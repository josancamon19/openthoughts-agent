"""
OpenThoughts Agent Datasets Dashboard
"""

import streamlit as st
from datasets import load_dataset

st.set_page_config(
    page_title="OpenThoughts Agent",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    
    .stApp {
        background: #0a0a0f;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #e2e8f0 !important;
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 600;
        color: #22d3ee;
        margin-bottom: 0.25rem;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    
    .stat-box {
        background: #111118;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #22d3ee;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .msg-user {
        background: #0c1a2e;
        border-left: 3px solid #3b82f6;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .msg-assistant {
        background: #1a0c1f;
        border-left: 3px solid #a855f7;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .msg-system {
        background: #0c1f1a;
        border-left: 3px solid #10b981;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .msg-role {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
        opacity: 0.7;
    }
    
    div[data-testid="stDataFrame"] {
        background: #111118;
        border-radius: 8px;
        padding: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading SFT dataset from HuggingFace...")
def load_sft_data():
    ds = load_dataset("open-thoughts/OpenThoughts-Agent-v1-SFT", split="train")
    return ds


@st.cache_data(show_spinner="Building index...")
def build_task_index(_ds):
    """Build task index with metadata."""
    tasks = {}

    for i in range(len(_ds)):
        row = _ds[i]
        task_id = row.get("task", "")
        conv = row.get("conversations", [])

        # Get preview from first user message and count assistant chars
        preview = ""
        assistant_chars = 0
        for msg in conv:
            if msg.get("role") == "assistant":
                assistant_chars += len(msg.get("content", ""))
            if msg.get("role") == "user" and not preview:
                content = msg.get("content", "")
                if "## Goal" in content:
                    goal_idx = content.find("## Goal")
                    goal_text = content[goal_idx + 8 : goal_idx + 150].strip()
                    preview = goal_text.replace("\n", " ").strip()
                elif "## Project Information" in content:
                    proj_idx = content.find("**Project:**")
                    if proj_idx != -1:
                        proj_text = content[proj_idx + 12 : proj_idx + 80].strip()
                        preview = f"Bug fix: {proj_text.split(chr(10))[0]}"
                else:
                    preview = content[:100].replace("\n", " ").strip()

        # Determine task type
        if task_id.startswith("task_"):
            task_type = "nl2bash"
        elif task_id.startswith("inferredbugs-"):
            task_type = "inferredbugs"
        else:
            task_type = "other"

        tasks[task_id] = {
            "ds_idx": i,
            "task_id": task_id,
            "msgs": len(conv),
            "preview": preview,
            "assistant_tokens": assistant_chars // 4,
            "task_type": task_type,
        }

    # Sort tasks: first by type (task_ before inferredbugs), then by number
    def task_sort_key(task_id):
        if task_id.startswith("task_"):
            try:
                return (0, int(task_id.split("_")[1]))
            except (IndexError, ValueError):
                return (0, 0)
        elif task_id.startswith("inferredbugs-"):
            try:
                return (1, int(task_id.split("-")[1]))
            except (IndexError, ValueError):
                return (1, 0)
        return (2, 0)

    sorted_task_ids = sorted(tasks.keys(), key=task_sort_key)

    return tasks, sorted_task_ids


def main():
    st.markdown(
        '<div class="main-title">🧠 OpenThoughts-Agent-v1-SFT</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">~15,200 conversation traces for supervised fine-tuning<br>'
        '<span style="font-size: 0.75rem; opacity: 0.7;">All traces generated with terminus-2 harness + QuantTrio/GLM-4.6-AWQ model</span></div>',
        unsafe_allow_html=True,
    )

    ds = load_sft_data()
    tasks, sorted_task_ids = build_task_index(ds)

    # Stats row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
        <div class="stat-box">
            <div class="stat-value">{len(ds):,}</div>
            <div class="stat-label">Total Traces</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="stat-box">
            <div class="stat-value">{len(tasks):,}</div>
            <div class="stat-label">Unique Tasks</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        nl2bash_count = sum(1 for t in sorted_task_ids if t.startswith("task_"))
        inferredbugs_count = sum(1 for t in sorted_task_ids if t.startswith("inferredbugs-"))
        st.markdown(
            f"""
        <div class="stat-box">
            <div class="stat-value">{nl2bash_count:,} / {inferredbugs_count:,}</div>
            <div class="stat-label">nl2bash / InferredBugs</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        task_type_filter = st.selectbox(
            "Task Type",
            ["All", "nl2bash", "inferredbugs"],
            index=0,
        )
    with filter_col2:
        sort_by = st.selectbox(
            "Sort by",
            ["Task ID", "Msgs ↑", "Msgs ↓", "Tokens ↑", "Tokens ↓"],
            index=0,
        )

    # Apply filters
    filtered_task_ids = []
    for task_id in sorted_task_ids:
        task = tasks[task_id]

        if task_type_filter != "All" and task["task_type"] != task_type_filter:
            continue

        filtered_task_ids.append(task_id)

    # Apply sorting
    if sort_by == "Msgs ↑":
        filtered_task_ids.sort(key=lambda t: tasks[t]["msgs"])
    elif sort_by == "Msgs ↓":
        filtered_task_ids.sort(key=lambda t: tasks[t]["msgs"], reverse=True)
    elif sort_by == "Tokens ↑":
        filtered_task_ids.sort(key=lambda t: tasks[t]["assistant_tokens"])
    elif sort_by == "Tokens ↓":
        filtered_task_ids.sort(key=lambda t: tasks[t]["assistant_tokens"], reverse=True)

    # Pagination
    page_size = 50
    total_pages = max(1, (len(filtered_task_ids) + page_size - 1) // page_size)

    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(filtered_task_ids))

    st.caption(f"Showing tasks {start_idx + 1} to {end_idx} of {len(filtered_task_ids):,}")

    # Build table for current page
    table_data = []
    for i in range(start_idx, end_idx):
        task_id = filtered_task_ids[i]
        task = tasks[task_id]

        table_data.append({
            "table_idx": i,
            "task_id": task_id,
            "msgs": task["msgs"],
            "tokens": task["assistant_tokens"],
            "preview": task["preview"][:250] + ("..." if len(task["preview"]) >= 250 else ""),
        })

    import pandas as pd
    df = pd.DataFrame(table_data)

    event = st.dataframe(
        df,
        column_config={
            "table_idx": None,
            "task_id": st.column_config.Column("Task ID", width=120),
            "msgs": st.column_config.Column("Msgs", width=60),
            "tokens": st.column_config.Column("Asst Tokens", width=90),
            "preview": st.column_config.Column("Preview", width=1200),
        },
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    # Show selected task details
    if event.selection and event.selection.rows:
        selected_row = event.selection.rows[0]
        task_id = table_data[selected_row]["task_id"]
        task = tasks[task_id]

        st.divider()
        st.subheader(f"Task: {task_id}")
        st.caption(f"{task['msgs']} messages | ~{task['assistant_tokens']:,} assistant tokens")

        sample = ds[task["ds_idx"]]

        # Show metadata
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Metadata**")
            for key, value in sample.items():
                if key != "conversations":
                    st.code(f"{key}: {value}", language=None)

        with col2:
            st.markdown("**Conversation**")
            conversations = sample.get("conversations", [])

            is_first_user = True
            for msg in conversations:
                role = msg.get("role", "unknown")
                content = msg.get("content", "").strip()

                if role == "user":
                    css_class = "msg-user"
                    icon = "👤"
                elif role == "assistant":
                    css_class = "msg-assistant"
                    icon = "🤖"
                else:
                    css_class = "msg-system"
                    icon = "⚙️"

                st.markdown(
                    f'<div class="{css_class}"><div class="msg-role">{icon} {role}</div></div>',
                    unsafe_allow_html=True,
                )

                # Split first user message into prefix and task
                if role == "user" and is_first_user and "Task Description:" in content:
                    is_first_user = False
                    split_idx = content.find("Task Description:")
                    prefix = content[:split_idx].strip()
                    task_content = content[split_idx:].strip()

                    with st.expander("📋 Shared Prefix (boilerplate)", expanded=False):
                        st.code(prefix, language=None)
                    st.markdown("**📌 Actual Task:**")
                    st.code(task_content, language=None)
                else:
                    if role == "user":
                        is_first_user = False
                    if len(content) > 2000:
                        with st.expander(f"View full content ({len(content):,} chars)"):
                            st.code(content, language=None)
                    else:
                        st.code(content, language=None)


if __name__ == "__main__":
    main()
