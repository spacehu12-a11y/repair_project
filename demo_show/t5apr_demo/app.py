import os
import re
import json
import time
import shutil
import difflib
import subprocess
from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd
import streamlit as st


# =========================================================
# Common helpers
# =========================================================
APP_VERSION = "2025-12-31"


def get_t5apr_home() -> Path:
    home = os.environ.get("T5APR_HOME", "").strip()
    if not home:
        raise RuntimeError("未设置环境变量 T5APR_HOME")
    p = Path(home).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"T5APR_HOME 路径不存在: {p}")
    return p


def unified_diff(a: str, b: str, fromfile: str, tofile: str) -> str:
    diff = difflib.unified_diff(
        a.splitlines(True),
        b.splitlines(True),
        fromfile=fromfile,
        tofile=tofile,
        lineterm="",
    )
    return "".join(diff)


def safe_read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def safe_write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def list_candidate_files(out_dir: Path) -> list[tuple[int, Path]]:
    files = []
    for p in out_dir.glob("final_candidates_*.jsonl"):
        m = re.match(r"final_candidates_(\d+)\.jsonl$", p.name)
        if m:
            files.append((int(m.group(1)), p))
    return sorted(files, key=lambda x: x[0])


@st.cache_data(show_spinner=False)
def read_jsonl_df(path: str) -> pd.DataFrame:
    p = Path(path)
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def run_subprocess(
    cmd: List[str], cwd: Path, timeout_s: int = 60
) -> Tuple[str, str, str]:
    """
    返回 (status, stdout, stderr)
    status ∈ {"PASS","FAIL","TIMEOUT","ERROR"}
    """
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        out, err = p.stdout, p.stderr
        if p.returncode == 0:
            return "PASS", out, err
        return "FAIL", out, err
    except subprocess.TimeoutExpired as e:
        return "TIMEOUT", (e.stdout or ""), (e.stderr or "")
    except Exception as e:
        return "ERROR", "", str(e)


def which(cmd: str) -> Optional[str]:
    try:
        r = subprocess.run(
            ["bash", "-lc", f"command -v {cmd}"], capture_output=True, text=True
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


# =========================================================
# Paths
# =========================================================
def outputs_dir_quixbugs_py(t5apr_home: Path, multi: bool) -> Path:
    gen_root = t5apr_home / "generated_assets" / "QuixBugs-Python"
    return gen_root / ("outputs-multi" if multi else "outputs-python")


def outputs_dir_quixbugs_java(t5apr_home: Path, multi: bool) -> Path:
    gen_root = t5apr_home / "generated_assets" / "QuixBugs-Java"
    return gen_root / ("outputs-multi" if multi else "outputs-java")


def quixbugs_root(t5apr_home: Path) -> Path:
    qb_root = t5apr_home / "benchmarks" / "QuixBugs"
    if not qb_root.exists():
        raise RuntimeError(f"未找到 QuixBugs 目录：{qb_root}")
    return qb_root


# =========================================================
# QuixBugs-Python ops
# =========================================================
def find_quixbugs_py_dirs(qb_root: Path) -> tuple[Path, Path]:
    candidates_programs = [
        qb_root / "python_programs",
        qb_root / "python_programs_buggy",
        qb_root / "python_programs_original",
    ]
    programs_dir = next((p for p in candidates_programs if p.exists()), None)
    if programs_dir is None:
        py_dirs = [
            d
            for d in qb_root.rglob("*")
            if d.is_dir() and d.name.lower().startswith("python_")
        ]
        py_dirs = [d for d in py_dirs if any(d.glob("*.py"))]
        programs_dir = py_dirs[0] if py_dirs else None
    if programs_dir is None:
        raise RuntimeError(
            "未定位到 QuixBugs 的 python_programs 目录（请检查 benchmarks/QuixBugs 结构）"
        )

    testcases_dir = qb_root / "python_testcases"
    if not testcases_dir.exists():
        tc = [d for d in qb_root.rglob("python_testcases") if d.is_dir()]
        if tc:
            testcases_dir = tc[0]
        else:
            raise RuntimeError("未定位到 QuixBugs 的 python_testcases 目录")

    return programs_dir, testcases_dir


def apply_patch_replace_first(
    src_text: str, source_snip: str, patch_snip: str
) -> tuple[str, bool]:
    if source_snip and (source_snip in src_text):
        return src_text.replace(source_snip, patch_snip, 1), True
    return src_text, False


def run_pytest_single(test_file: Path, cwd: Path, timeout_s: int = 60) -> tuple[str, str, str]:
    cmd = ["python", "-m", "pytest", "-x", str(test_file)]
    return run_subprocess(cmd, cwd=cwd, timeout_s=timeout_s)


def fast_oracle_signal(row: pd.Series) -> bool:
    """
    “疑似可修复”的快速信号（不等于真正通过测试）：
    - exact_match / correct 为 True
    - 或 decoded_sequences == target
    """
    for k in ["exact_match", "correct"]:
        if k in row and bool(row[k]):
            return True
    if "decoded_sequences" in row and "target" in row:
        return str(row["decoded_sequences"]).strip() == str(row["target"]).strip()
    return False


# =========================================================
# QuixBugs-Java ops
# =========================================================
def find_quixbugs_java_dirs(qb_root: Path) -> tuple[Path, Path]:
    programs = qb_root / "java_programs"
    if not programs.exists():
        raise RuntimeError("未找到 QuixBugs 的 java_programs 目录")
    tests = qb_root / "java_testcases" / "junit"
    if not tests.exists():
        tc = [d for d in qb_root.rglob("java_testcases") if d.is_dir()]
        if tc:
            tests = tc[0]
        else:
            raise RuntimeError("未定位到 QuixBugs 的 java_testcases 目录")
    return programs, tests


def run_gradle_compile_and_test(project_dir: Path, bugid: str, timeout_s: int = 120) -> tuple[str, str, str]:
    gradlew = project_dir / "gradlew"
    if gradlew.exists():
        gradle_cmd = ["./gradlew"]
    elif which("gradle"):
        gradle_cmd = ["gradle"]
    else:
        return "ERROR", "", "未找到 gradle 或 gradlew，无法进行 Java 验证。"

    # build w/o test
    comp = gradle_cmd + ["build", "-x", "test", "-p", str(project_dir)]
    status, out, err = run_subprocess(comp, cwd=project_dir, timeout_s=timeout_s)
    if status != "PASS":
        return "FAIL", out, err

    # run one test class (QuixBugs often uses <BUGID>_TEST)
    test_name = f"{bugid.upper()}_TEST"
    test_cmd = gradle_cmd + ["test", "-p", str(project_dir), "--tests", test_name]
    return run_subprocess(test_cmd, cwd=project_dir, timeout_s=timeout_s)


# =========================================================
# Custom file (online inference) helpers
# =========================================================
def build_model_input(prefix: str, removed: str, context: str, eos_token: str, unk_token: str) -> str:
    source = f"{prefix} {removed.strip()} :"
    ctx = " ".join(context.split())
    s = f"{source} {ctx}"
    if eos_token:
        s = s.replace(eos_token, unk_token or "")
    return s


def extract_context_window(lines: list[str], bug_line_1based: int, bug_len: int, window: int) -> tuple[str, str, str]:
    """
    返回 (removed_text, context_text, indent)
    removed_text：bug 行段
    context_text：上下文（去掉 removed 段）
    indent：removed 第一行缩进
    """
    n = len(lines)
    start0 = max(0, bug_line_1based - 1)
    end0 = min(n, start0 + max(1, bug_len))
    removed = "".join(lines[start0:end0]).rstrip("\n")
    indent = re.match(r"^\s*", lines[start0]).group(0) if start0 < n else ""

    left0 = max(0, start0 - window)
    right0 = min(n, end0 + window)
    ctx_lines = lines[left0:start0] + lines[end0:right0]
    context = " ".join("".join(ctx_lines).split())
    return removed, context, indent


@st.cache_resource(show_spinner=False)
def load_local_model(checkpoint_dir: str):
    import torch  # lazy
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tok = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    return tok, model, device


def generate_topk_patches_local(
    checkpoint_dir: Path,
    model_input: str,
    topk: int = 10,
    max_target_len: int = 256,
) -> list[tuple[str, float]]:
    import torch  # lazy

    tok, model, device = load_local_model(str(checkpoint_dir))
    enc = tok(
        [model_input],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            **enc,
            num_beams=topk,
            num_return_sequences=topk,
            max_length=max_target_len,
            min_length=0,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
    seqs = tok.batch_decode(out.sequences, skip_special_tokens=True)
    scores = (
        out.sequences_scores.detach().to("cpu").tolist()
        if hasattr(out, "sequences_scores")
        else [0.0] * len(seqs)
    )
    pairs = list(zip(seqs, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def py_compile_check(file_path: Path) -> tuple[str, str, str]:
    cmd = ["python", "-m", "py_compile", str(file_path)]
    return run_subprocess(cmd, cwd=file_path.parent, timeout_s=30)


def run_pytest_custom(test_paths: list[Path], cwd: Path, timeout_s: int) -> tuple[str, str, str]:
    cmd = ["python", "-m", "pytest", "-x"] + [str(p) for p in test_paths]
    return run_subprocess(cmd, cwd=cwd, timeout_s=timeout_s)


# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="T5APR 展示", layout="wide")
st.title("T5APR（深度学习）补丁生成与验证展示")
st.caption(f"App version: {APP_VERSION}")

with st.sidebar:
    st.header("全局配置")
    mode = st.radio(
        "切换展示模块",
        ["QuixBugs-Python", "QuixBugs-Java", "自定义文件（Python）"],
        index=0,
    )
    multi = st.checkbox("使用 multi（outputs-multi / 多语言模型）", value=True)
    timeout_s = st.number_input("测试超时（秒）", min_value=10, max_value=600, value=60, step=10)

try:
    t5apr_home = get_t5apr_home()
except Exception as e:
    st.error(str(e))
    st.stop()

st.caption(f"T5APR_HOME：`{t5apr_home}`")


# -----------------------------
# Page: QuixBugs-Python
# -----------------------------
if mode == "QuixBugs-Python":
    out_dir = outputs_dir_quixbugs_py(t5apr_home, multi=multi)
    st.subheader("QuixBugs-Python：候选补丁展示 + 单 bug 自动验证")
    st.write(f"输出目录：`{out_dir}`")

    cand_files = list_candidate_files(out_dir)
    if not cand_files:
        st.error("未找到 final_candidates_*.jsonl。请先生成 candidates 并 combine。")
        st.stop()

    with st.sidebar:
        ks = [k for k, _ in cand_files]
        k = st.selectbox("选择候选文件（k）", ks, index=len(ks) - 1)

    cand_path = dict(cand_files)[k]
    df = read_jsonl_df(str(cand_path))

    if "bugid" not in df.columns:
        st.error("候选文件缺少 bugid 字段，无法展示。")
        st.stop()

    st.markdown(
        """
**字段说明（写报告/答辩用）**  
- **候选数**：该 bugid 在候选文件中一共有多少条候选补丁。  
- **疑似可修复**：快速信号（`exact_match` 或 `correct` 或 `decoded_sequences == target` 之一成立）。  
  该信号 **不等于** 真正修复成功；真正成功以 **pytest PASS** 为准。
        """.strip()
    )

    # 计算疑似可修复
    df["_oracle_like"] = df.apply(fast_oracle_signal, axis=1)

    summary = (
        df.groupby("bugid")
        .agg(
            候选数=("bugid", "size"),
            疑似可修复=("_oracle_like", "any"),
        )
        .reset_index()
    )

    with st.sidebar:
        only_oracle = st.checkbox("只看“疑似可修复”的 bugid", value=False)

    bugids = (
        summary.sort_values(["疑似可修复", "bugid"], ascending=[False, True])["bugid"]
        .tolist()
    )
    if only_oracle:
        bugids = summary.loc[summary["疑似可修复"], "bugid"].tolist()

    if not bugids:
        st.warning("没有可展示的 bugid（可能你筛选了“疑似可修复”，但当前文件没有）。")
        st.stop()

    with st.sidebar:
        default_bug = "bitcount" if "bitcount" in bugids else bugids[0]
        bugid = st.selectbox(
            "选择 bugid",
            bugids,
            index=bugids.index(default_bug) if default_bug in bugids else 0,
        )
        auto_topn = st.slider(
            "自动验证 Top-N（找到 PASS 即停止）",
            1,
            min(20, int(k)),
            value=min(5, int(k)),
        )

    st.markdown("#### 数据集概览（快速定位“疑似可修复样例”）")
    st.dataframe(
        summary.sort_values(["疑似可修复", "bugid"], ascending=[False, True]),
        use_container_width=True,
    )

    df_bug = df[df["bugid"] == bugid].copy()
    if "rank" in df_bug.columns:
        df_bug = df_bug.sort_values(["hunk", "rank"], ascending=[True, True])
    elif "sequences_scores" in df_bug.columns:
        df_bug = df_bug.sort_values(["hunk", "sequences_scores"], ascending=[True, False])

    st.success(f"读取：`{cand_path}`；bugid=`{bugid}`；候选数={len(df_bug)}")

    show_cols = [
        c
        for c in [
            "hunk",
            "rank",
            "checkpoint",
            "decoded_sequences",
            "sequences_scores",
            "source",
            "target",
            "exact_match",
            "correct",
        ]
        if c in df_bug.columns
    ]
    st.dataframe(df_bug[show_cols].head(50), use_container_width=True)

    st.divider()
    st.subheader("验证区（pytest）")

    default_run_id = f"qbpy_{bugid}_{int(time.time())}"
    run_id = st.text_input("run_id（隔离并发/中断污染）", value=default_run_id).strip()

    st.markdown("##### 方式 A：手动选择一条候选补丁验证")
    options = []
    topN_df = df_bug.head(20)
    for i, row in topN_df.iterrows():
        patch = str(row.get("decoded_sequences", ""))
        score = row.get("sequences_scores", "")
        rk = row.get("rank", "")
        options.append((i, f"idx={i} | rank={rk} | patch={patch[:60]} | score={score}"))
    sel = st.radio("选择候选（Top 20）", options, format_func=lambda x: x[1])
    sel_idx = sel[0]
    sel_row = df_bug.loc[sel_idx]

    st.markdown("##### 方式 B：自动验证 Top-N（推荐演示用）")
    colA, colB = st.columns([1, 1])
    with colA:
        run_manual = st.button("验证：仅当前选择", type="primary")
    with colB:
        run_auto = st.button(f"自动验证 Top-{auto_topn}", type="secondary")

    def do_one_candidate(row: pd.Series, attempt_name: str) -> dict:
        qb_root = quixbugs_root(t5apr_home)
        programs_dir, testcases_dir = find_quixbugs_py_dirs(qb_root)

        runs_root = out_dir / "runs" / run_id
        work_qb = runs_root / "QuixBugs"
        work_programs = work_qb / programs_dir.name
        work_tests = work_qb / testcases_dir.name

        attempt_dir = runs_root / "attempts" / attempt_name
        if attempt_dir.exists():
            shutil.rmtree(attempt_dir, ignore_errors=True)
        attempt_dir.mkdir(parents=True, exist_ok=True)

        # 每次都重拷贝一份，避免污染
        if work_qb.exists():
            shutil.rmtree(work_qb, ignore_errors=True)
        shutil.copytree(qb_root, work_qb, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".*"))

        src_file = work_programs / f"{bugid}.py"
        test_file = work_tests / f"test_{bugid}.py"
        if not src_file.exists():
            raise RuntimeError(f"未找到 buggy 源文件：{src_file}")
        if not test_file.exists():
            raise RuntimeError(f"未找到测试文件：{test_file}")

        src_text = safe_read_text(src_file)
        source_snip = str(row.get("source", ""))
        patch_snip = str(row.get("decoded_sequences", ""))

        new_text, ok = apply_patch_replace_first(src_text, source_snip, patch_snip)
        if not ok:
            ns = str(row.get("normalized_source", ""))
            npat = str(row.get("normalized_patch", ""))
            if ns and npat:
                new_text, ok = apply_patch_replace_first(src_text, ns, npat)

        if not ok:
            return {
                "status": "ERROR",
                "msg": "未匹配到 source/normalized_source，无法应用补丁（可换候选）",
                "attempt_dir": str(attempt_dir),
                "diff": "",
                "stdout": "",
                "stderr": "",
            }

        safe_write_text(src_file, new_text)
        diff = unified_diff(src_text, new_text, "buggy", "patched")
        safe_write_text(attempt_dir / "patched.py", new_text)
        safe_write_text(attempt_dir / "diff.patch", diff)

        status, out, err = run_pytest_single(test_file=test_file, cwd=work_qb, timeout_s=int(timeout_s))
        safe_write_text(attempt_dir / "pytest.out.txt", out or "")
        safe_write_text(attempt_dir / "pytest.err.txt", err or "")

        return {
            "status": status,
            "msg": "",
            "attempt_dir": str(attempt_dir),
            "diff": diff,
            "stdout": out,
            "stderr": err,
        }

    def show_result(res: dict, ok_label: str = "PASS") -> None:
        st.code(res["diff"] or "(无 diff)", language="diff")
        if res["status"] == ok_label:
            st.success(f"{ok_label}：找到 plausible patch")
        elif res["status"] == "FAIL":
            st.warning("FAIL：未通过测试")
        elif res["status"] == "TIMEOUT":
            st.warning("TIMEOUT：测试超时")
        else:
            st.error(f"ERROR：{res.get('msg','')}")
        st.info(f"落盘目录：`{res['attempt_dir']}`")
        with st.expander("stdout"):
            st.text(res.get("stdout") or "(empty)")
        with st.expander("stderr"):
            st.text(res.get("stderr") or "(empty)")

    if run_manual:
        attempt_name = f"idx_{sel_idx}_rank_{sel_row.get('rank','')}"
        with st.spinner("正在应用补丁并运行 pytest..."):
            res = do_one_candidate(sel_row, attempt_name=attempt_name)
        show_result(res)

    if run_auto:
        st.write(f"将按 rank/score 顺序验证 Top-{auto_topn}，遇到 PASS 立刻停止。")
        prog = st.progress(0)
        first_pass = None
        log_rows = []

        candidates = df_bug.head(int(auto_topn)).to_dict(orient="records")
        for i, row_dict in enumerate(candidates, start=1):
            row = pd.Series(row_dict)
            attempt_name = f"auto_{i}_rank_{row.get('rank','')}"
            with st.spinner(f"验证 {i}/{auto_topn} ..."):
                res = do_one_candidate(row, attempt_name=attempt_name)
            log_rows.append(
                {
                    "i": i,
                    "rank": row.get("rank", ""),
                    "score": row.get("sequences_scores", ""),
                    "patch": str(row.get("decoded_sequences", ""))[:80],
                    "status": res["status"],
                    "attempt_dir": res["attempt_dir"],
                }
            )
            prog.progress(int(i * 100 / auto_topn))
            if res["status"] == "PASS":
                first_pass = (row, res)
                break

        st.dataframe(pd.DataFrame(log_rows), use_container_width=True)

        if first_pass:
            row, res = first_pass
            st.success(f"PASS：已找到 plausible patch（rank={row.get('rank','')}）")
            st.code(res["diff"] or "(无 diff)", language="diff")
            st.info(f"落盘目录：`{res['attempt_dir']}`")
        else:
            st.warning(f"Top-{auto_topn} 未找到 PASS（你可以增大 Top-N 或换 bugid）")


# -----------------------------
# Page: QuixBugs-Java
# -----------------------------
elif mode == "QuixBugs-Java":
    out_dir = outputs_dir_quixbugs_java(t5apr_home, multi=multi)
    st.subheader("QuixBugs-Java：候选补丁展示 + 自动验证（Gradle）")
    st.write(f"输出目录：`{out_dir}`")

    cand_files = list_candidate_files(out_dir)
    if not cand_files:
        st.warning("未找到 final_candidates_*.jsonl（你还没生成 Java 的 candidates 或未 combine）。")
        st.stop()

    with st.sidebar:
        ks = [k for k, _ in cand_files]
        k = st.selectbox("选择候选文件（k）", ks, index=len(ks) - 1)

    cand_path = dict(cand_files)[k]
    df = read_jsonl_df(str(cand_path))

    if "bugid" not in df.columns:
        st.error("候选文件缺少 bugid 字段。")
        st.stop()

    bugids = sorted(df["bugid"].unique().tolist())
    with st.sidebar:
        bugid = st.selectbox("选择 bugid", bugids, index=0)
        java_timeout = st.number_input("Java 验证超时（秒）", 30, 600, 180, 10)
        auto_topn = st.slider("自动验证 Top-N", 1, min(20, int(k)), value=min(5, int(k)))

    qb_root = quixbugs_root(t5apr_home)
    st.caption(f"Gradle 检测：gradlew={'有' if (qb_root/'gradlew').exists() else '无'}；gradle={'有' if which('gradle') else '无'}")

    df_bug = df[df["bugid"] == bugid].copy()
    if "rank" in df_bug.columns:
        df_bug = df_bug.sort_values(["hunk", "rank"], ascending=[True, True])
    elif "sequences_scores" in df_bug.columns:
        df_bug = df_bug.sort_values(["hunk", "sequences_scores"], ascending=[True, False])

    show_cols = [
        c
        for c in [
            "hunk",
            "rank",
            "checkpoint",
            "decoded_sequences",
            "sequences_scores",
            "source",
            "target",
            "exact_match",
            "correct",
        ]
        if c in df_bug.columns
    ]
    st.dataframe(df_bug[show_cols].head(50), use_container_width=True)

    st.divider()
    st.subheader("验证区（Gradle：build + 单测）")

    default_run_id = f"qbjava_{bugid}_{int(time.time())}"
    run_id = st.text_input("run_id（隔离并发/中断污染）", value=default_run_id).strip()

    st.markdown("##### 方式 A：手动选择一条候选补丁验证")
    options = []
    topN_df = df_bug.head(20)
    for i, row in topN_df.iterrows():
        patch = str(row.get("decoded_sequences", ""))
        score = row.get("sequences_scores", "")
        rk = row.get("rank", "")
        options.append((i, f"idx={i} | rank={rk} | patch={patch[:60]} | score={score}"))
    sel = st.radio("选择候选（Top 20）", options, format_func=lambda x: x[1])
    sel_idx = sel[0]
    sel_row = df_bug.loc[sel_idx]

    st.markdown("##### 方式 B：自动验证 Top-N（推荐演示用）")
    colA, colB = st.columns([1, 1])
    with colA:
        run_manual = st.button("验证：仅当前选择（Java）", type="primary")
    with colB:
        run_auto = st.button(f"自动验证 Top-{auto_topn}（Java）", type="secondary")

    def do_one_java_candidate(row: pd.Series, attempt_name: str) -> dict:
        qb_root0 = quixbugs_root(t5apr_home)

        runs_root = out_dir / "runs" / run_id
        work_qb = runs_root / "QuixBugs"
        attempt_dir = runs_root / "attempts" / attempt_name
        if attempt_dir.exists():
            shutil.rmtree(attempt_dir, ignore_errors=True)
        attempt_dir.mkdir(parents=True, exist_ok=True)

        # 重拷贝
        if work_qb.exists():
            shutil.rmtree(work_qb, ignore_errors=True)
        shutil.copytree(qb_root0, work_qb, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".*"))

        # QuixBugs Java 文件名通常是 BUGID 大写
        src_file = work_qb / "java_programs" / f"{bugid.upper()}.java"
        if not src_file.exists():
            return {
                "status": "ERROR",
                "msg": f"未找到 Java 源文件：{src_file}",
                "attempt_dir": str(attempt_dir),
                "diff": "",
                "stdout": "",
                "stderr": "",
            }

        src_text = safe_read_text(src_file)
        source_snip = str(row.get("source", ""))
        patch_snip = str(row.get("decoded_sequences", ""))
        new_text, ok = apply_patch_replace_first(src_text, source_snip, patch_snip)
        if not ok:
            ns = str(row.get("normalized_source", ""))
            npat = str(row.get("normalized_patch", ""))
            if ns and npat:
                new_text, ok = apply_patch_replace_first(src_text, ns, npat)

        if not ok:
            return {
                "status": "ERROR",
                "msg": "未匹配到 source/normalized_source，无法应用补丁（可换候选）",
                "attempt_dir": str(attempt_dir),
                "diff": "",
                "stdout": "",
                "stderr": "",
            }

        safe_write_text(src_file, new_text)
        diff = unified_diff(src_text, new_text, "buggy", "patched")
        safe_write_text(attempt_dir / "patched.java", new_text)
        safe_write_text(attempt_dir / "diff.patch", diff)

        status, out, err = run_gradle_compile_and_test(work_qb, bugid=bugid, timeout_s=int(java_timeout))
        safe_write_text(attempt_dir / "gradle.out.txt", out or "")
        safe_write_text(attempt_dir / "gradle.err.txt", err or "")

        return {
            "status": status,
            "msg": "" if status != "ERROR" else (err or "gradle error"),
            "attempt_dir": str(attempt_dir),
            "diff": diff,
            "stdout": out,
            "stderr": err,
        }

    def show_java_result(res: dict) -> None:
        st.code(res["diff"] or "(无 diff)", language="diff")
        if res["status"] == "PASS":
            st.success("PASS：Java 编译 + 单测通过（plausible patch）")
        elif res["status"] == "FAIL":
            st.warning("FAIL：编译或测试失败")
        elif res["status"] == "TIMEOUT":
            st.warning("TIMEOUT：超时")
        else:
            st.error(f"ERROR：{res.get('msg','')}")
        st.info(f"落盘目录：`{res['attempt_dir']}`")
        with st.expander("gradle stdout"):
            st.text(res.get("stdout") or "(empty)")
        with st.expander("gradle stderr"):
            st.text(res.get("stderr") or "(empty)")

    if run_manual:
        attempt_name = f"idx_{sel_idx}_rank_{sel_row.get('rank','')}"
        with st.spinner("正在应用补丁并运行 Gradle..."):
            res = do_one_java_candidate(sel_row, attempt_name=attempt_name)
        show_java_result(res)

    if run_auto:
        st.write(f"将按 rank/score 顺序验证 Top-{auto_topn}，遇到 PASS 立刻停止。")
        prog = st.progress(0)
        first_pass = None
        log_rows = []

        candidates = df_bug.head(int(auto_topn)).to_dict(orient="records")
        for i, row_dict in enumerate(candidates, start=1):
            row = pd.Series(row_dict)
            attempt_name = f"auto_{i}_rank_{row.get('rank','')}"
            with st.spinner(f"验证 {i}/{auto_topn} ..."):
                res = do_one_java_candidate(row, attempt_name=attempt_name)
            log_rows.append(
                {
                    "i": i,
                    "rank": row.get("rank", ""),
                    "score": row.get("sequences_scores", ""),
                    "patch": str(row.get("decoded_sequences", ""))[:80],
                    "status": res["status"],
                    "attempt_dir": res["attempt_dir"],
                }
            )
            prog.progress(int(i * 100 / auto_topn))
            if res["status"] == "PASS":
                first_pass = (row, res)
                break

        st.dataframe(pd.DataFrame(log_rows), use_container_width=True)

        if first_pass:
            row, res = first_pass
            st.success(f"PASS：已找到 plausible patch（rank={row.get('rank','')}）")
            st.code(res["diff"] or "(无 diff)", language="diff")
            st.info(f"落盘目录：`{res['attempt_dir']}`")
        else:
            st.warning(f"Top-{auto_topn} 未找到 PASS（可增大 Top-N 或换 bugid）")


# -----------------------------
# Page: Custom Python file
# -----------------------------
else:
    st.subheader("自定义文件（Python）：在线生成候选补丁 + 检测 + 展示")
    st.caption("说明：需要你提供错误位置（行号/长度），本模块不做自动定位。")

    out_dir = outputs_dir_quixbugs_py(t5apr_home, multi=multi)  # runs 放在 outputs-multi 下
    runs_root = out_dir / "runs"

    # 本地模型
    models_root = t5apr_home / "models"
    model_name = "codet5-small-t5apr-multi" if multi else "codet5-small-t5apr-python"
    checkpoints_dir = models_root / model_name
    if not checkpoints_dir.exists():
        st.error(f"未找到模型目录：{checkpoints_dir}（请确认已下载 checkpoints）")
        st.stop()

    ckpts = []
    for d in checkpoints_dir.iterdir():
        if d.is_dir() and re.match(r"checkpoint-\d+$", d.name):
            ckpts.append(d)
    ckpts.sort(key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        st.error(f"未找到 checkpoint-*：{checkpoints_dir}")
        st.stop()

    with st.sidebar:
        st.header("自定义生成配置")
        ckpt = st.selectbox("选择 checkpoint", [c.name for c in ckpts], index=len(ckpts) - 1)
        topk = st.slider("Top-K 候选补丁数", 1, 20, 10, 1)
        ctx_win = st.slider("上下文窗口（行）", 1, 50, 15, 1)
        max_target_len = st.slider("max_target_len", 32, 512, 256, 32)
        do_pytest = st.checkbox("同时运行 pytest（你需要上传测试文件）", value=False)

    ckpt_dir = checkpoints_dir / ckpt

    uploaded = st.file_uploader("上传待修复的 .py 文件", type=["py"])
    if not uploaded:
        st.stop()

    code_text = uploaded.getvalue().decode("utf-8", errors="ignore")
    st.markdown("#### 原始代码（预览）")
    st.code(code_text, language="python")

    test_uploads = []
    if do_pytest:
        test_uploads = st.file_uploader(
            "（可选）上传 pytest 测试文件（一个或多个 .py）",
            type=["py"],
            accept_multiple_files=True,
        )

    with st.sidebar:
        bug_line = st.number_input("错误起始行（1-based）", min_value=1, value=1, step=1)
        bug_len = st.number_input("错误行数（>=1）", min_value=1, value=1, step=1)
        default_run_id = f"custom_{Path(uploaded.name).stem}_{int(time.time())}"
        run_id = st.text_input("run_id", value=default_run_id).strip()

    if st.button("生成候选补丁（Top-K）", type="primary"):
        try:
            lines = code_text.splitlines(True)
            removed, context, indent = extract_context_window(
                lines, int(bug_line), int(bug_len), int(ctx_win)
            )

            try:
                tok, _, _ = load_local_model(str(ckpt_dir))
            except ModuleNotFoundError as e:
                st.error(f"缺少依赖：{e}。请在当前 Streamlit 虚拟环境安装 torch/transformers。")
                st.stop()

            model_input = build_model_input(
                prefix="Python",
                removed=removed,
                context=context,
                eos_token=getattr(tok, "eos_token", "") or "",
                unk_token=getattr(tok, "unk_token", "") or "",
            )

            st.markdown("#### 模型输入（答辩解释用）")
            st.code(model_input, language="text")

            with st.spinner("正在生成候选补丁..."):
                pairs = generate_topk_patches_local(
                    checkpoint_dir=ckpt_dir,
                    model_input=model_input,
                    topk=int(topk),
                    max_target_len=int(max_target_len),
                )

            # 去重（保留最高分）
            seen = {}
            for p, s in pairs:
                p2 = p.strip()
                if not p2:
                    continue
                if p2 not in seen or s > seen[p2]:
                    seen[p2] = s
            pairs = sorted(seen.items(), key=lambda x: x[1], reverse=True)

            st.session_state["custom_pairs"] = pairs
            st.session_state["custom_removed"] = removed
            st.success(f"生成候选数：{len(pairs)}（已去重）")

        except Exception as e:
            st.error(str(e))

    pairs = st.session_state.get("custom_pairs")
    if not pairs:
        st.stop()

    st.markdown("#### 候选补丁列表（按 score 排序）")
    cand_df = pd.DataFrame(
        [{"rank": i, "patch": p, "score": s} for i, (p, s) in enumerate(pairs)]
    )
    st.dataframe(cand_df, use_container_width=True)

    sel_rank = st.number_input(
        "选择 rank",
        min_value=0,
        max_value=len(pairs) - 1,
        value=0,
        step=1,
    )
    chosen_patch, chosen_score = pairs[int(sel_rank)]

    def apply_by_line_replace(original: str, line1: int, length: int, patch: str) -> str:
        L = original.splitlines(True)
        start0 = max(0, line1 - 1)
        end0 = min(len(L), start0 + max(1, length))
        indent = re.match(r"^\s*", L[start0]).group(0) if start0 < len(L) else ""
        repl = indent + patch.strip("\n") + "\n"
        L[start0:end0] = [repl]
        return "".join(L)

    patched_text = apply_by_line_replace(
        code_text, int(bug_line), int(bug_len), chosen_patch
    )
    st.markdown("#### 补丁 diff")
    st.code(unified_diff(code_text, patched_text, "buggy", "patched") or "(无 diff)", language="diff")

    if st.button("落盘并检测（py_compile / 可选 pytest）", type="secondary"):
        run_dir = runs_root / run_id / "custom"
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(uploaded.name).stem
        buggy_path = run_dir / f"{stem}.buggy.py"
        patched_path = run_dir / f"{stem}.patched.py"
        safe_write_text(buggy_path, code_text)
        safe_write_text(patched_path, patched_text)
        safe_write_text(run_dir / "diff.patch", unified_diff(code_text, patched_text, "buggy", "patched"))

        # 1) py_compile
        status, out, err = py_compile_check(patched_path)
        safe_write_text(run_dir / "py_compile.out.txt", out or "")
        safe_write_text(run_dir / "py_compile.err.txt", err or "")

        if status == "PASS":
            st.success("py_compile PASS：语法/字节码编译通过（基础可运行性 OK）")
        elif status == "FAIL":
            st.warning("py_compile FAIL：仍有语法错误/编译错误")
        else:
            st.error(f"py_compile ERROR：{err}")

        # 2) optional pytest
        if do_pytest:
            if not test_uploads:
                st.warning("你勾选了 pytest，但没有上传测试文件；已跳过 pytest。")
            else:
                tests_dir = run_dir / "tests"
                tests_dir.mkdir(parents=True, exist_ok=True)
                test_paths = []
                for f in test_uploads:
                    p = tests_dir / f.name
                    p.write_bytes(f.getvalue())
                    test_paths.append(p)

                st.write("运行 pytest ...")
                pstat, pout, perr = run_pytest_custom(test_paths=test_paths, cwd=run_dir, timeout_s=int(timeout_s))
                safe_write_text(run_dir / "pytest.out.txt", pout or "")
                safe_write_text(run_dir / "pytest.err.txt", perr or "")

                if pstat == "PASS":
                    st.success("pytest PASS：测试通过（更强的正确性信号）")
                elif pstat == "FAIL":
                    st.warning("pytest FAIL：测试失败")
                elif pstat == "TIMEOUT":
                    st.warning("pytest TIMEOUT：超时")
                else:
                    st.error(f"pytest ERROR：{perr}")

                with st.expander("pytest stdout"):
                    st.text(pout or "(empty)")
                with st.expander("pytest stderr"):
                    st.text(perr or "(empty)")

        st.info(f"落盘目录：`{run_dir}`")

        st.download_button(
            "下载修复后的文件（patched.py）",
            data=patched_text.encode("utf-8"),
            file_name=f"{stem}.patched.py",
            mime="text/x-python",
        )
