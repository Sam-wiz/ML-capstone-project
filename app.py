"""
app.py — Streamlit UI for the Job Application Assistant.
Run with: streamlit run app.py
"""

import streamlit as st
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()

# ── DB + Auth bootstrap (must happen before graph import) ─────────────────────
from db.database import init_db
init_db()

from auth.auth import login, register, decode_token
from db.database import (
    get_user_by_id, get_applications,
    add_application, update_application_status, delete_application,
)
from graph import graph
from state import HumanFeedback, GraphState
from config import has_openai_api_key
from agents.job_finder import find_jobs
from agents.interview_prep import run_interview_prep, CATEGORIES
from tools.pdf_parser import parse_resume_pdf
from tools.resume_pdf_writer import (
    generate_resume_structured,
    render_resume_html,
    render_resume_pdf_from_data,
)

_is_authed = bool(st.session_state.get("user") or st.session_state.get("jwt_token"))
st.set_page_config(
    page_title="Job Application Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded" if _is_authed else "collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
}
input, textarea, .stTextInput input, .stTextArea textarea,
div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {
    background-color: #ffffff !important; color: #0f172a !important;
    border: 1.5px solid #e2e8f0 !important; border-radius: 10px !important;
    font-size: 14px !important;
}
input:focus, textarea:focus,
div[data-baseweb="input"] input:focus,
div[data-baseweb="textarea"] textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}
input::placeholder, textarea::placeholder { color: #94a3b8 !important; }
label, .stTextInput label, .stTextArea label {
    color: #374151 !important; font-size: 13px !important;
    font-weight: 600 !important;
}
div[data-baseweb="select"] > div {
    background-color: #ffffff !important; border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important; color: #0f172a !important;
}
div.stButton > button {
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    font-size: 14px !important; border-radius: 10px !important;
    border: none !important; padding: 10px 20px !important;
    transition: all 0.15s ease !important;
}
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important; box-shadow: 0 2px 8px rgba(37,99,235,0.28) !important;
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    transform: translateY(-1px) !important;
}
div.stButton > button[kind="secondary"],
div.stButton > button[data-testid="baseButton-secondary"] {
    background: #ffffff !important; color: #374151 !important;
    border: 1.5px solid #e2e8f0 !important;
}
div.stDownloadButton > button {
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    border-radius: 10px !important;
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important; border: none !important;
}
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #2563eb, #60a5fa) !important;
    border-radius: 999px !important;
}
div[data-testid="stProgressBar"] > div {
    background: #e2e8f0 !important; border-radius: 999px !important; height: 6px !important;
}
div[data-testid="stAlert"] { border-radius: 10px !important; font-size: 14px !important; }
details { background: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 10px !important; }
details summary { font-weight: 600 !important; font-size: 14px !important; color: #374151 !important; }
.stTabs [data-baseweb="tab-list"] {
    background: #e2e8f0 !important; border-radius: 12px !important;
    padding: 4px !important; gap: 2px !important; border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 9px !important;
    font-weight: 600 !important; font-size: 13px !important;
    color: #64748b !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important; color: #1d4ed8 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10) !important;
}
section[data-testid="stSidebar"] { background: #0f172a !important; border-right: 1px solid #1e293b !important; }
section[data-testid="stSidebar"] *:not(button) { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSuccess {
    background: rgba(22,163,74,0.15) !important; border-color: rgba(22,163,74,0.4) !important;
    color: #86efac !important; border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stWarning {
    background: rgba(217,119,6,0.15) !important; color: #fcd34d !important; border-radius: 8px !important;
}
section[data-testid="stSidebar"] div.stButton > button {
    background: rgba(255,255,255,0.08) !important; color: #cbd5e1 !important;
    border: 1px solid rgba(255,255,255,0.12) !important; width: 100% !important;
}

.page-hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 60%, #3b82f6 100%);
    border-radius: 16px; padding: 32px 36px; margin-bottom: 28px;
    position: relative; overflow: hidden;
}
.page-hero h1 { color:#fff !important; font-size:24px !important; font-weight:800 !important; margin:0 0 6px 0 !important; }
.page-hero p  { color:#bfdbfe !important; font-size:14px !important; margin:0 !important; }

.card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: box-shadow 0.18s ease, transform 0.18s ease;
}
.card:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.08); transform: translateY(-1px); }
.job-title { font-size:17px; font-weight:700; color:#0f172a; margin-bottom:3px; }
.job-meta  { font-size:12.5px; color:#64748b; margin-bottom:8px; }

.src-badge-gh  { display:inline-block; background:#dcfce7; color:#15803d;
                 font-size:11px; font-weight:700; padding:2px 9px; border-radius:999px; margin-right:6px; }
.src-badge-ddg { display:inline-block; background:#eff6ff; color:#1d4ed8;
                 font-size:11px; font-weight:700; padding:2px 9px; border-radius:999px; margin-right:6px; }

.section-label {
    font-size:10.5px; font-weight:700; text-transform:uppercase;
    letter-spacing:0.09em; color:#94a3b8; margin-bottom:8px; margin-top:20px;
}

.score-wrap {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:16px;
    padding:24px 18px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.05);
}
.score-num  { font-size:56px; font-weight:900; line-height:1; letter-spacing:-0.04em; }
.score-sub  { font-size:11px; color:#94a3b8; margin-top:2px; font-weight:500; }
.score-rec  { font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; margin-top:6px; }
.c-green { color:#16a34a; } .c-amber { color:#d97706; } .c-red { color:#dc2626; }

.kw-chip { display:inline-block; font-size:12px; font-weight:600; padding:3px 10px; border-radius:999px; margin:2px; }
.kw-matched { background:#eff6ff; color:#1d4ed8; }
.kw-added   { background:#fef9c3; color:#854d0e; }

.step-track { display:flex; align-items:center; background:#ffffff; border:1px solid #e2e8f0;
              border-radius:12px; padding:12px 18px; margin-bottom:24px; }
.step-item  { display:flex; align-items:center; gap:7px; flex:1; font-size:13px; font-weight:500; color:#94a3b8; }
.step-item.s-done   { color:#16a34a; }
.step-item.s-active { color:#1d4ed8; font-weight:700; }
.step-pip   { width:24px; height:24px; border-radius:50%; display:flex; align-items:center;
              justify-content:center; font-size:11px; font-weight:700; flex-shrink:0;
              background:#e2e8f0; color:#94a3b8; }
.s-done   .step-pip { background:#dcfce7; color:#16a34a; }
.s-active .step-pip { background:#2563eb; color:#ffffff; }
.step-line      { flex:0 0 24px; height:2px; background:#e2e8f0; }
.step-line.l-done { background:#86efac; }

.tracker-card { background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
                padding:16px 20px; margin-bottom:8px; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
.status-pill  { display:inline-block; font-size:11px; font-weight:700; padding:3px 10px; border-radius:999px; }
.s-Applied    { background:#eff6ff; color:#1d4ed8; }
.s-Screening  { background:#fef9c3; color:#854d0e; }
.s-Interview  { background:#ede9fe; color:#6d28d9; }
.s-Offer      { background:#dcfce7; color:#15803d; }
.s-Rejected   { background:#fee2e2; color:#dc2626; }

.iq-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:14px 18px; margin-bottom:10px; }
.iq-q    { font-weight:700; font-size:14px; color:#0f172a; margin-bottom:6px; }
.iq-cat  { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; color:#94a3b8; }
.iq-tip  { font-size:12px; color:#64748b; font-style:italic; margin-top:6px; }
.iq-tp   { font-size:13px; color:#374151; margin:2px 0 2px 10px; }

.final-wrap { background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
              padding:28px 32px; box-shadow:0 1px 3px rgba(0,0,0,0.05);
              font-size:14px; line-height:1.7; }

.auth-wrap { max-width:420px; margin:60px auto; background:#ffffff;
             border:1px solid #e2e8f0; border-radius:16px; padding:36px;
             box-shadow:0 4px 20px rgba(0,0,0,0.08); }
.auth-title { font-size:22px; font-weight:800; color:#0f172a; margin-bottom:4px; }
.auth-sub   { font-size:14px; color:#64748b; margin-bottom:24px; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
_D = {
    "jwt_token": None,
    "user": None,
    "thread_id": f"st-{uuid4()}",
    "graph_state": None,
    "stage": "input",
    "final_package": None,
    "page": "finder",
    "jobs_found": [],
    "selected_job": None,
    "jd_input": "",
    "resume_input": "",
    "company_input": "",
    "role_input": "",
    "resume_pdf_bytes": None,
    "resume_structured_data": None,
    "interview_prep_data": None,
}
for k, v in _D.items():
    if k not in st.session_state:
        st.session_state[k] = v

api_ok = has_openai_api_key()


# ── Auth helpers ──────────────────────────────────────────────────────────────
def _load_user_from_token():
    token = st.session_state.jwt_token
    if not token:
        return
    payload = decode_token(token)
    if not payload:
        st.session_state.jwt_token = None
        st.session_state.user = None
        return
    if not st.session_state.user:
        user = get_user_by_id(payload["sub"])
        st.session_state.user = user


def _logout():
    st.session_state.jwt_token = None
    st.session_state.user = None
    st.rerun()


_load_user_from_token()
tc = {"configurable": {"thread_id": st.session_state.thread_id}}


# ── Auth page ─────────────────────────────────────────────────────────────────
if not st.session_state.user:
    # mode toggle stored in session to avoid tabs-inside-columns rendering issues
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "signin"

    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## 💼 Job Application Assistant")
        st.markdown("*AI-powered resume tailoring, job search, and application tracking.*")
        st.markdown("---")

        mode = st.session_state.auth_mode

        # Mode toggle
        c1, c2 = st.columns(2)
        if c1.button("Sign In",       use_container_width=True,
                     type="primary" if mode=="signin"   else "secondary", key="mode_si"):
            st.session_state.auth_mode = "signin";  st.rerun()
        if c2.button("Create Account", use_container_width=True,
                     type="primary" if mode=="register" else "secondary", key="mode_reg"):
            st.session_state.auth_mode = "register"; st.rerun()

        st.markdown("")

        if mode == "signin":
            email_l = st.text_input("Email", key="li_email", placeholder="you@example.com")
            pass_l  = st.text_input("Password", key="li_pass", type="password")
            if st.button("Sign In →", type="primary", use_container_width=True, key="li_btn"):
                if not email_l.strip() or not pass_l:
                    st.error("Please fill in both fields.")
                else:
                    token, msg = login(email_l.strip(), pass_l)
                    if token:
                        st.session_state.jwt_token = token
                        st.session_state.auth_mode = "signin"
                        st.rerun()
                    else:
                        st.error(msg)

        else:  # register
            name_r  = st.text_input("Full name",  key="reg_name")
            email_r = st.text_input("Email",      key="reg_email", placeholder="you@example.com")
            pass_r  = st.text_input("Password",   key="reg_pass", type="password",
                                    help="Minimum 6 characters")
            if st.button("Create Account →", type="primary", use_container_width=True, key="reg_btn"):
                if not name_r.strip() or not email_r.strip() or not pass_r:
                    st.error("Please fill in all fields.")
                else:
                    ok, msg = register(name_r.strip(), email_r.strip(), pass_r)
                    if ok:
                        token, _ = login(email_r.strip(), pass_r)
                        if token:
                            st.session_state.jwt_token = token
                            st.session_state.auth_mode = "signin"
                            st.rerun()
                    else:
                        st.error(msg)

    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
user = st.session_state.user
with st.sidebar:
    st.markdown(f"### 💼 Job Assistant")
    st.caption(f"Signed in as **{user['name'] or user['email']}**")
    st.markdown("---")

    nav = st.radio("", ["🔍 Job Finder", "✏️ Tailor Application", "📋 My Applications"],
                   index={"finder":0,"tailor":1,"tracker":2}.get(st.session_state.page, 0),
                   label_visibility="collapsed")
    target = {"🔍 Job Finder":"finder","✏️ Tailor Application":"tailor","📋 My Applications":"tracker"}[nav]
    if target != st.session_state.page:
        st.session_state.page = target
        st.rerun()

    st.markdown("---")
    if api_ok:
        st.success("✓ OpenAI API ready")
    else:
        st.warning("⚠ Set OPENAI_API_KEY in .env")

    st.markdown("---")
    if st.button("↺  Reset session", use_container_width=True):
        for k in ["thread_id","graph_state","stage","final_package","jobs_found",
                  "selected_job","jd_input","resume_input","company_input","role_input",
                  "resume_pdf_bytes","resume_structured_data","interview_prep_data"]:
            if k in st.session_state: del st.session_state[k]
        st.rerun()
    if st.button("🚪  Sign Out", use_container_width=True):
        _logout()

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("LangGraph · LangChain · Greenhouse · OpenAI")


# ── Step tracker ──────────────────────────────────────────────────────────────
def _steps(current):
    items = [("input","1","Input"),("running","2","Agents"),("review","3","Review"),("done","✓","Done")]
    stages = [s for s,_,_ in items]
    try: idx = stages.index(current)
    except ValueError: idx = 0
    parts = []
    for i,(s,icon,label) in enumerate(items):
        cls = "s-done" if i < idx else ("s-active" if i == idx else "")
        parts.append(f'<div class="step-item {cls}"><div class="step-pip">{icon}</div><span>{label}</span></div>')
        if i < len(items)-1:
            lc = "l-done" if i < idx else ""
            parts.append(f'<div class="step-line {lc}"></div>')
    st.markdown(f'<div class="step-track">{"".join(parts)}</div>', unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
# JOB FINDER
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "finder":
    st.markdown("""<div class="page-hero">
        <h1>🔍 Job Finder</h1>
        <p>Live Greenhouse listings + DuckDuckGo search — apply directly to Greenhouse roles.</p>
    </div>""", unsafe_allow_html=True)

    col_q, col_r = st.columns(2, gap="large")
    with col_q:
        st.markdown('<div class="section-label">Search</div>', unsafe_allow_html=True)
        query    = st.text_input("Keywords or job title", placeholder="e.g. Backend Python Developer", label_visibility="collapsed")
        location = st.text_input("Location", placeholder="e.g. Remote, London", label_visibility="collapsed")
    with col_r:
        st.markdown('<div class="section-label">Match with resume (optional)</div>', unsafe_allow_html=True)
        pdf_up = st.file_uploader("Upload PDF resume", type=["pdf"], key="finder_pdf", label_visibility="collapsed")
        if pdf_up is not None:
            try:
                raw_bytes = pdf_up.read()
                parsed = parse_resume_pdf(raw_bytes)
                st.session_state.resume_input = parsed
                st.session_state.resume_pdf_bytes = raw_bytes
                st.success(f"✓ Parsed {pdf_up.name} ({len(parsed.split())} words)")
            except Exception as e:
                st.error(f"PDF parse failed: {e}")
        resume_match = st.text_area("Resume", height=90, value=st.session_state.resume_input,
                                    placeholder="Upload PDF above or paste text…", label_visibility="collapsed")
        if resume_match != st.session_state.resume_input:
            st.session_state.resume_input = resume_match

    st.markdown("")
    if st.button("🔍  Find Jobs", type="primary", use_container_width=True):
        if not query.strip():
            st.error("Enter a keyword to search.")
        elif not api_ok:
            st.error("Configure OPENAI_API_KEY in .env first.")
        else:
            q = query.strip() + (f" in {location.strip()}" if location.strip() else "")
            with st.spinner("Searching Greenhouse + DuckDuckGo…"):
                try:
                    st.session_state.jobs_found = find_jobs(q, st.session_state.resume_input)
                    gh = sum(1 for j in st.session_state.jobs_found if j["source"]=="greenhouse")
                    ddg = len(st.session_state.jobs_found) - gh
                    st.toast(f"{gh} Greenhouse + {ddg} DuckDuckGo listings", icon="💼")
                except Exception as e:
                    st.error(f"Search failed: {e}")

    if st.session_state.jobs_found:
        gh_count  = sum(1 for j in st.session_state.jobs_found if j["source"]=="greenhouse")
        ddg_count = len(st.session_state.jobs_found) - gh_count
        st.markdown(
            f'<div class="section-label">{len(st.session_state.jobs_found)} positions — '
            f'<span class="src-badge-gh">✓ {gh_count} Greenhouse</span>'
            f'<span class="src-badge-ddg">⚡ {ddg_count} DuckDuckGo</span></div>',
            unsafe_allow_html=True,
        )

        for job in st.session_state.jobs_found:
            is_gh = job["source"] == "greenhouse"
            badge = (
                '<span class="src-badge-gh">✓ Greenhouse — Live Listing</span>'
                if is_gh else
                '<span class="src-badge-ddg">⚡ DuckDuckGo</span>'
            )
            st.markdown(f"""<div class="card">
                <div class="job-title">{job['title']}</div>
                <div class="job-meta">🏢 {job['company']} &nbsp;·&nbsp; 📍 {job['location']} &nbsp;·&nbsp; 💰 {job.get('salary','N/A')}</div>
                {badge}
            </div>""", unsafe_allow_html=True)

            with st.expander(f"Details — {job['title']} at {job['company']}"):
                st.write(job["description"][:600] + "…")
                if job.get("requirements"):
                    c1, c2 = st.columns(2)
                    for i, r in enumerate(job["requirements"]):
                        (c1 if i%2==0 else c2).markdown(f"- {r}")
                if is_gh and job.get("apply_url"):
                    st.markdown(f"🔗 [View on Greenhouse]({job['apply_url']})", unsafe_allow_html=False)

            if st.button("🎯 Tailor for this role", key=f"tailor_{job['id']}", type="primary"):
                st.session_state.selected_job  = job
                st.session_state.jd_input      = job["description"]
                st.session_state.company_input = job["company"]
                st.session_state.role_input    = job["title"]
                st.session_state.page          = "tailor"
                st.session_state.stage         = "input"
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION TRACKER
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "tracker":
    st.markdown("""<div class="page-hero">
        <h1>📋 My Applications</h1>
        <p>Track every role you've applied to — update status as you progress.</p>
    </div>""", unsafe_allow_html=True)

    apps = get_applications(user["id"])
    statuses = ["Applied","Screening","Interview","Offer","Rejected"]

    # ── summary counts ──────────────────────────────────────────────────────
    if apps:
        counts = {s: sum(1 for a in apps if a["status"]==s) for s in statuses}
        cols = st.columns(5)
        pill_cls = {"Applied":"kw-matched","Screening":"kw-added","Interview":"","Offer":"c-green","Rejected":"c-red"}
        for i, s in enumerate(statuses):
            cols[i].metric(s, counts[s])
        st.markdown("---")

    # ── add manual entry ────────────────────────────────────────────────────
    with st.expander("➕ Add manual entry"):
        m1, m2 = st.columns(2)
        mt = m1.text_input("Job Title", key="mt")
        mc = m2.text_input("Company",   key="mc")
        ml = st.text_input("Location",  key="ml")
        mn = st.text_area("Notes",      key="mn", height=60)
        if st.button("Add", key="add_manual"):
            if mt and mc:
                add_application(user["id"], mt, mc, ml, "manual", notes=mn)
                st.success("Added.")
                st.rerun()

    if not apps:
        st.info("No applications yet. Apply to a Greenhouse job or add one manually above.")
    else:
        for app in apps:
            src_badge = (
                '<span class="src-badge-gh">✓ Greenhouse</span>'
                if app["source"]=="greenhouse" else
                f'<span style="font-size:11px;color:#94a3b8;">{app["source"]}</span>'
            )
            st.markdown(f"""<div class="tracker-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                  <div>
                    <strong style="font-size:15px;">{app['job_title']}</strong>
                    <span style="color:#64748b;font-size:13px;"> · {app['company']}</span>
                    {'<br><span style="font-size:12px;color:#94a3b8;">📍 '+app['location']+'</span>' if app['location'] else ''}
                  </div>
                  <div style="text-align:right;">
                    <span class="status-pill s-{app['status']}">{app['status']}</span><br>
                    <span style="font-size:11px;color:#94a3b8;">{app['applied_at'][:10]}</span>
                  </div>
                </div>
                <div style="margin-top:6px;">{src_badge}</div>
            </div>""", unsafe_allow_html=True)

            ec1, ec2, ec3 = st.columns([3,2,1])
            new_status = ec1.selectbox(
                "Status", statuses,
                index=statuses.index(app["status"]),
                key=f"st_{app['id']}",
                label_visibility="collapsed",
            )
            new_note = ec2.text_input("Note", value=app.get("notes","") or "",
                                       key=f"nt_{app['id']}", label_visibility="collapsed",
                                       placeholder="Add a note…")
            c_save, c_del = ec3.columns(2)
            if c_save.button("✓", key=f"sv_{app['id']}", help="Save"):
                update_application_status(app["id"], user["id"], new_status, new_note)
                st.rerun()
            if c_del.button("🗑", key=f"dl_{app['id']}", help="Delete"):
                delete_application(app["id"], user["id"])
                st.rerun()
            st.markdown("")


# ═══════════════════════════════════════════════════════════════════════════════
# TAILOR APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "tailor":
    st.markdown("""<div class="page-hero">
        <h1>✏️ Tailor Application</h1>
        <p>AI agents tailor your CV and cover letter — you review before anything is finalised.</p>
    </div>""", unsafe_allow_html=True)

    _steps(st.session_state.stage)

    if st.session_state.selected_job:
        job = st.session_state.selected_job
        c1, c2 = st.columns([6,1])
        c1.info(f"🎯 Tailoring for **{job['title']}** at **{job['company']}**"
                + (" · Greenhouse ✓" if job.get("source")=="greenhouse" else ""))
        if c2.button("Clear"):
            for k in ["selected_job","jd_input","company_input","role_input"]:
                st.session_state[k] = None if k=="selected_job" else ""
            st.rerun()

    # ── INPUT ────────────────────────────────────────────────────────────────
    if st.session_state.stage == "input":
        L, R = st.columns(2, gap="large")
        with L:
            st.markdown('<div class="section-label">Job details</div>', unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            rv = m1.text_input("Role", value=st.session_state.role_input)
            if rv != st.session_state.role_input: st.session_state.role_input = rv
            cv = m2.text_input("Company", value=st.session_state.company_input)
            if cv != st.session_state.company_input: st.session_state.company_input = cv
            jd = st.text_area("Job description", value=st.session_state.jd_input, height=300,
                               placeholder="Paste the full job description…")
            if jd != st.session_state.jd_input: st.session_state.jd_input = jd
        with R:
            st.markdown('<div class="section-label">Your resume</div>', unsafe_allow_html=True)
            pdf_up2 = st.file_uploader("Upload PDF resume", type=["pdf"], key="tailor_pdf", label_visibility="collapsed")
            if pdf_up2 is not None:
                try:
                    raw_bytes2 = pdf_up2.read()
                    parsed = parse_resume_pdf(raw_bytes2)
                    st.session_state.resume_input = parsed
                    st.session_state.resume_pdf_bytes = raw_bytes2
                    st.success(f"✓ Parsed {pdf_up2.name} ({len(parsed.split())} words)")
                except Exception as e:
                    st.error(f"PDF parse failed: {e}")
            res = st.text_area("Resume", value=st.session_state.resume_input, height=310,
                                placeholder="Upload PDF above or paste plain text…")
            if res != st.session_state.resume_input: st.session_state.resume_input = res

        st.markdown("")
        if st.button("🚀  Run Pipeline", type="primary", use_container_width=True):
            if not st.session_state.jd_input.strip() or not st.session_state.resume_input.strip():
                st.error("Both job description and resume are required.")
            elif not api_ok:
                st.error("Configure OPENAI_API_KEY in .env.")
            else:
                init: GraphState = {
                    "job_description": st.session_state.jd_input,
                    "raw_resume": st.session_state.resume_input,
                    "company_name": st.session_state.company_input,
                    "role_title": st.session_state.role_input,
                    "guardrail_passed": False, "guardrail_message": "",
                    "fit_score": None, "tailored_cv": None, "cover_letter": None,
                    "human_feedback": None, "awaiting_human": False,
                    "routing_decision": None, "final_package": None,
                    "agent_log": [], "error": None,
                }
                st.session_state.thread_id = f"st-{uuid4()}"
                tc = {"configurable": {"thread_id": st.session_state.thread_id}}
                with st.spinner("Running agents… (20–40 s)"):
                    evs = list(graph.stream(init, tc, stream_mode="values"))
                    st.session_state.graph_state = evs[-1] if evs else init
                s = st.session_state.graph_state
                if not s.get("guardrail_passed"):
                    st.error(f"❌ {s.get('guardrail_message')}")
                else:
                    snap = graph.get_state(tc)
                    st.session_state.stage = (
                        "low_fit" if snap.next and "low_fit_warning" in str(snap.next)
                        else "review"
                    )
                    st.rerun()

    # ── LOW FIT ──────────────────────────────────────────────────────────────
    elif st.session_state.stage == "low_fit":
        s   = st.session_state.graph_state
        fit = s.get("fit_score", {})
        st.warning(f"**Low Fit — Score: {fit.get('score',0)}/100.** This role may be a stretch.")
        g_col, s_col = st.columns(2)
        with g_col:
            st.markdown("**Gaps**")
            for g in fit.get("gaps", []): st.markdown(f"- {g}")
        with s_col:
            st.markdown("**Strengths**")
            for s2 in fit.get("strengths", []): st.markdown(f"- {s2}")
        ca, cb = st.columns(2)
        if ca.button("✅  Proceed anyway", type="primary", use_container_width=True):
            graph.update_state(tc, {"human_feedback": HumanFeedback(decision="approve").model_dump(), "awaiting_human": False})
            with st.spinner("Continuing…"):
                evs = list(graph.stream(None, tc, stream_mode="values"))
                st.session_state.graph_state = evs[-1] if evs else st.session_state.graph_state
            st.session_state.stage = "review"; st.rerun()
        if cb.button("❌  Abort", use_container_width=True):
            graph.update_state(tc, {"human_feedback": HumanFeedback(decision="reject").model_dump(), "awaiting_human": False})
            list(graph.stream(None, tc, stream_mode="values"))
            st.session_state.stage = "aborted"; st.rerun()

    # ── REVIEW (HITL) ─────────────────────────────────────────────────────────
    elif st.session_state.stage == "review":
        s   = st.session_state.graph_state
        fit = s.get("fit_score", {})
        cv  = s.get("tailored_cv", {})
        cl  = s.get("cover_letter", {})
        score = fit.get("score", 0)
        rec   = fit.get("recommendation", "")
        sc    = "c-green" if score >= 70 else ("c-amber" if score >= 40 else "c-red")

        left, right = st.columns([1,3], gap="large")
        with left:
            st.markdown(f"""<div class="score-wrap">
                <div class="score-num {sc}">{score}</div>
                <div class="score-sub">out of 100</div>
                <div class="score-rec {sc}">{rec.replace('_',' ').title()}</div>
            </div>""", unsafe_allow_html=True)
            st.progress(score/100)
            kw = fit.get("keyword_analysis", {})
            matched = kw.get("matched_keywords", [])
            added   = kw.get("missing_keywords", [])
            if matched:
                st.markdown('<div class="section-label">Matched</div>', unsafe_allow_html=True)
                st.markdown("".join(f'<span class="kw-chip kw-matched">{k}</span>' for k in matched[:8]), unsafe_allow_html=True)
            if added:
                st.markdown('<div class="section-label">Added by AI</div>', unsafe_allow_html=True)
                st.markdown("".join(f'<span class="kw-chip kw-added">{k}</span>' for k in added[:8]), unsafe_allow_html=True)

        with right:
            t_fit, t_cv, t_cl, t_log = st.tabs(["📊 Fit Analysis","📄 Tailored CV","✉️ Cover Letter","🔍 Agent Log"])
            with t_fit:
                fa, fb = st.columns(2)
                with fa:
                    st.markdown("**Strengths**")
                    for s2 in fit.get("strengths",[]): st.markdown(f"✅ {s2}")
                with fb:
                    st.markdown("**Gaps**")
                    for g in fit.get("gaps",[]): st.markdown(f"⚠️ {g}")
                if fit.get("reasoning"): st.markdown(f"_{fit['reasoning']}_")
            with t_cv:
                e_summary = st.text_area("Professional Summary", value=cv.get("summary",""), height=90)
                st.markdown('<div class="section-label">Experience bullets</div>', unsafe_allow_html=True)
                e_bullets = st.text_area("Bullets", value="\n".join(cv.get("tailored_bullets",[])),
                                          height=180, label_visibility="collapsed")
                e_skills  = st.text_input("Skills (comma-separated)", value=", ".join(cv.get("skills_section",[])))
            with t_cl:
                if cl.get("subject_line"): st.markdown(f"**Subject:** {cl['subject_line']}")
                e_cl = st.text_area("Cover letter", value=cl.get("full_text",""), height=340, label_visibility="collapsed")
            with t_log:
                for entry in s.get("agent_log",[]): st.code(entry, language=None)

        st.markdown("---")
        notes = st.text_input("Optional notes", placeholder="e.g. Soften the opening paragraph")
        ca2, cb2, cc2 = st.columns(3)
        with ca2:
            if st.button("✅  Approve & Generate", type="primary", use_container_width=True):
                graph.update_state(tc, {"human_feedback": HumanFeedback(decision="approve", feedback_notes=notes or None).model_dump(), "awaiting_human": False})
                with st.spinner("Assembling…"):
                    evs = list(graph.stream(None, tc, stream_mode="values"))
                    st.session_state.graph_state = evs[-1] if evs else s
                st.session_state.final_package = st.session_state.graph_state.get("final_package")
                # auto-track if from a known job
                if st.session_state.selected_job:
                    job = st.session_state.selected_job
                    add_application(
                        user_id           = user["id"],
                        job_title         = job.get("title", st.session_state.role_input),
                        company           = job.get("company", st.session_state.company_input),
                        location          = job.get("location",""),
                        source            = job.get("source","tailor"),
                        greenhouse_board  = job.get("greenhouse_board",""),
                        greenhouse_job_id = job.get("greenhouse_job_id",""),
                        fit_score         = score,
                        job_description   = st.session_state.jd_input[:500],
                    )
                st.session_state.stage = "done"; st.rerun()
        with cb2:
            if st.button("✏️  Save Edits & Approve", use_container_width=True):
                ecv = (f"**Professional Summary**\n{e_summary}\n\n**Experience:**\n"
                       + "\n".join(f"- {b}" for b in e_bullets.split("\n") if b.strip())
                       + f"\n\n**Skills:** {e_skills}")
                graph.update_state(tc, {"human_feedback": HumanFeedback(decision="edit", edited_cv=ecv,
                    edited_cover_letter=e_cl, feedback_notes=notes or None).model_dump(), "awaiting_human": False})
                with st.spinner("Assembling with edits…"):
                    evs = list(graph.stream(None, tc, stream_mode="values"))
                    st.session_state.graph_state = evs[-1] if evs else s
                st.session_state.final_package = st.session_state.graph_state.get("final_package")
                st.session_state.stage = "done"; st.rerun()
        with cc2:
            if st.button("❌  Reject & Discard", use_container_width=True):
                graph.update_state(tc, {"human_feedback": HumanFeedback(decision="reject", feedback_notes=notes or None).model_dump(), "awaiting_human": False})
                list(graph.stream(None, tc, stream_mode="values"))
                st.session_state.stage = "aborted"; st.rerun()

    # ── DONE ─────────────────────────────────────────────────────────────────
    elif st.session_state.stage == "done":
        st.success("🎉 Application package ready — also added to My Applications.")

        gs          = st.session_state.graph_state or {}
        tailored_cv = gs.get("tailored_cv") or {}
        has_resume  = bool(tailored_cv and st.session_state.resume_input.strip())

        # Generate + cache structured resume data
        if has_resume and st.session_state.resume_structured_data is None:
            with st.spinner("Applying AI edits to your resume…"):
                try:
                    st.session_state.resume_structured_data = generate_resume_structured(
                        tailored_cv     = tailored_cv,
                        original_text   = st.session_state.resume_input,
                        role_title      = st.session_state.role_input,
                        company_name    = st.session_state.company_input,
                        job_description = st.session_state.jd_input,
                    )
                except Exception as e:
                    st.warning(f"Resume generation failed: {e}")

        # Generate + cache interview prep
        if st.session_state.interview_prep_data is None and st.session_state.jd_input and st.session_state.resume_input:
            with st.spinner("Generating interview prep…"):
                try:
                    st.session_state.interview_prep_data = run_interview_prep(
                        job_description = st.session_state.jd_input,
                        resume          = st.session_state.resume_input,
                        role_title      = st.session_state.role_input,
                        company         = st.session_state.company_input,
                    )
                except Exception as e:
                    st.warning(f"Interview prep failed: {e}")

        rdata   = st.session_state.resume_structured_data
        ip_data = st.session_state.interview_prep_data

        t_pkg, t_res, t_ip = st.tabs(["📋 Application Package", "📄 Updated Resume", "🎤 Interview Prep"])

        with t_pkg:
            st.markdown('<div class="final-wrap">', unsafe_allow_html=True)
            st.markdown(st.session_state.final_package or "_No output generated._")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("")
            st.download_button("⬇️  Download Package (Markdown)",
                data=st.session_state.final_package or "",
                file_name="application_package.md", mime="text/markdown",
                use_container_width=True, type="primary")

        with t_res:
            if rdata:
                st.markdown(render_resume_html(rdata), unsafe_allow_html=True)
                st.markdown("")
                try:
                    pdf_bytes = render_resume_pdf_from_data(rdata)
                    st.download_button("⬇️  Download Updated Resume (PDF)",
                        data=pdf_bytes, file_name="tailored_resume.pdf",
                        mime="application/pdf", use_container_width=True, type="primary")
                except Exception as e:
                    st.error(f"PDF render failed: {e}")
            else:
                st.info("Upload your resume PDF before running the pipeline to get a tailored PDF download.")

        with t_ip:
            if ip_data:
                questions = ip_data.get("questions", [])
                red_flags = ip_data.get("red_flags_to_avoid", [])

                # ── Summary bar ────────────────────────────────────────────────
                DIFF_COLOR = {"easy":"#16a34a","medium":"#d97706","hard":"#dc2626"}
                cat_counts = {}
                for q in questions:
                    c = CATEGORIES.get(q.get("category","").lower(), {}).get("label", q.get("category","").capitalize())
                    cat_counts[c] = cat_counts.get(c, 0) + 1

                badges = "".join(
                    f'<span style="display:inline-block;background:#f1f5f9;border:1px solid #e2e8f0;'
                    f'font-size:11px;font-weight:700;padding:3px 10px;border-radius:999px;margin:2px;">'
                    f'{label} · {count}</span>'
                    for label, count in cat_counts.items()
                )
                st.markdown(f"""<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
                    padding:18px 22px;margin-bottom:18px;">
                    <div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:10px;">
                        {len(questions)} Questions — {role_input or 'Role'} at {company_input or 'Company'}
                    </div>
                    <div>{badges}</div>
                    <div style="margin-top:12px;padding:12px;background:#eff6ff;border-radius:8px;
                         border-left:3px solid #2563eb;font-size:13px;color:#1e40af;font-style:italic;">
                        💡 {ip_data.get('overall_tip','')}
                    </div>
                </div>""", unsafe_allow_html=True)

                # ── Red flags ─────────────────────────────────────────────────
                if red_flags:
                    rf_html = "".join(
                        f'<div style="font-size:13px;color:#7f1d1d;padding:4px 0;">⚠️ {rf}</div>'
                        for rf in red_flags
                    )
                    st.markdown(f"""<div style="background:#fef2f2;border:1px solid #fecaca;
                        border-radius:10px;padding:14px 18px;margin-bottom:18px;">
                        <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                             letter-spacing:0.08em;color:#dc2626;margin-bottom:6px;">
                            Common mistakes to avoid
                        </div>
                        {rf_html}
                    </div>""", unsafe_allow_html=True)

                # ── Category filter ──────────────────────────────────────────
                all_cats = sorted(set(q.get("category","").lower() for q in questions))
                cat_labels = ["All"] + [
                    CATEGORIES.get(c, {}).get("label", c.capitalize()) for c in all_cats
                ]
                chosen = st.selectbox("Filter by category", cat_labels, key="ip_filter",
                                      label_visibility="collapsed")

                # ── Question cards ────────────────────────────────────────────
                role_input  = st.session_state.role_input
                company_input = st.session_state.company_input

                for i, q in enumerate(questions, 1):
                    cat   = q.get("category", "").lower()
                    meta  = CATEGORIES.get(cat, {"label": cat.capitalize(), "color": "#f8fafc",
                                                  "accent": "#64748b", "icon": "❓"})
                    label = meta["label"]

                    if chosen != "All" and chosen != label:
                        continue

                    diff      = q.get("difficulty", "medium").lower()
                    diff_col  = DIFF_COLOR.get(diff, "#64748b")
                    tps_html  = "".join(
                        f'<div style="font-size:13px;color:#374151;padding:3px 0 3px 14px;'
                        f'border-left:2px solid {meta["accent"]}33;">'
                        f'→ {tp}</div>'
                        for tp in q.get("talking_points", [])
                    )
                    structure = q.get("sample_answer_structure", "")
                    tip       = q.get("tip", "")

                    st.markdown(f"""<div style="background:{meta['color']};border:1px solid {meta['accent']}33;
                        border-left:4px solid {meta['accent']};border-radius:12px;
                        padding:16px 20px;margin-bottom:12px;">

                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                            <div>
                                <span style="font-size:11px;font-weight:800;text-transform:uppercase;
                                    letter-spacing:0.08em;color:{meta['accent']};">
                                    {meta['icon']} {label}
                                </span>
                                <span style="font-size:11px;font-weight:600;color:{diff_col};
                                    background:{diff_col}22;padding:1px 8px;border-radius:999px;
                                    margin-left:8px;">{diff.upper()}</span>
                            </div>
                            <span style="font-size:11px;color:#94a3b8;font-weight:700;">Q{i} / {len(questions)}</span>
                        </div>

                        <div style="font-size:15px;font-weight:700;color:#0f172a;
                            margin-bottom:10px;line-height:1.4;">
                            {q.get('question', '')}
                        </div>

                        <div style="margin-bottom:10px;">{tps_html}</div>

                        <div style="display:flex;gap:10px;flex-wrap:wrap;">
                            <div style="font-size:12px;background:#ffffff88;border-radius:7px;
                                padding:7px 12px;flex:1;min-width:160px;">
                                <span style="font-weight:700;color:#0f172a;">Structure: </span>
                                <span style="color:#374151;">{structure}</span>
                            </div>
                            <div style="font-size:12px;background:#ffffff88;border-radius:7px;
                                padding:7px 12px;flex:1;min-width:160px;">
                                <span style="font-weight:700;color:#0f172a;">Tip: </span>
                                <span style="color:#374151;font-style:italic;">{tip}</span>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("Interview prep questions will appear here after the pipeline runs.")

        st.markdown("")
        if st.button("🔄  Start New Application", use_container_width=True):
            for k in ["graph_state","stage","final_package","thread_id","selected_job",
                      "resume_pdf_bytes","resume_structured_data","interview_prep_data"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()

    # ── ABORTED ───────────────────────────────────────────────────────────────
    elif st.session_state.stage == "aborted":
        st.warning("Application aborted — nothing was saved.")
        if st.button("🔄  Start Over", use_container_width=True):
            for k in ["graph_state","stage","final_package","thread_id","selected_job",
                      "resume_pdf_bytes","resume_structured_data","interview_prep_data"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()
