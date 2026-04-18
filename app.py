import os, io, datetime
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from google import genai
from dotenv import load_dotenv

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image as RLImage, HRFlowable)

load_dotenv()

st.set_page_config(page_title="RadVision AI", page_icon="🩺",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');
:root{--bg:#050810;--bg-card:#0b0f1e;--bg-card2:#0f1526;--cyan:#00d4ff;--green:#00e887;
      --red:#ff3d57;--yellow:#ffcc00;--purple:#7c6dfa;--muted:#4b5675;
      --border:rgba(0,212,255,0.12);--text:#dce8f5;}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:var(--bg)!important;color:var(--text);}
.main,.block-container{background:var(--bg)!important;}
.block-container{padding:2.5rem 2.5rem 2rem!important;max-width:1400px!important;}
.topbar{display:flex;align-items:center;justify-content:space-between;
        padding:1.4rem 0 1.6rem;border-bottom:1px solid var(--border);margin-bottom:2.5rem;}
.logo{font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:800;letter-spacing:-.02em;
      background:linear-gradient(100deg,var(--cyan),var(--green));
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.logo-sub{-webkit-text-fill-color:var(--muted);font-weight:400;font-size:.82rem;
          margin-left:12px;font-family:'Inter',sans-serif;}
.badge-live{display:inline-flex;align-items:center;gap:6px;
            background:rgba(0,232,135,.08);border:1px solid rgba(0,232,135,.22);
            border-radius:50px;padding:5px 14px;font-size:.7rem;font-weight:600;
            color:var(--green);letter-spacing:.07em;}
.badge-live::before{content:'';width:6px;height:6px;border-radius:50%;
                    background:var(--green);animation:blink 1.8s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}
.panel-title{font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;
             text-align:center;letter-spacing:.05em;text-transform:uppercase;
             color:var(--text);margin-bottom:1.6rem;display:flex;align-items:center;gap:10px;}
.panel-title::after{content:'';flex:1;height:1px;background:var(--border);}
.stats-row{display:flex;gap:10px;margin-bottom:1.8rem;}
.stat-chip{flex:1;background:var(--bg-card2);border:1px solid var(--border);
           border-radius:10px;padding:.8rem 1rem;text-align:center;}
.stat-chip .sv{font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;color:var(--cyan);}
.stat-chip .sl{font-size:.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-top:3px;}
.upload-hint{text-align:center;padding:3rem 1rem;color:var(--muted);font-size:.85rem;line-height:2;}
.upload-hint .icon{font-size:2.8rem;margin-bottom:.4rem;display:block;}
.detection-result{border-radius:12px;padding:1.1rem 1.3rem;margin:1.2rem 0;
                  display:flex;align-items:center;gap:14px;}
.detection-result .d-icon{font-size:2rem;}
.detection-result .d-label{font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700;}
.detection-result .d-sub{font-size:.78rem;color:var(--muted);margin-top:3px;}
.result-normal{background:rgba(0,232,135,.07);border:1px solid rgba(0,232,135,.28);}
.result-cancer{background:rgba(255,61,87,.07);border:1px solid rgba(255,61,87,.28);}
.result-fracture{background:rgba(255,204,0,.07);border:1px solid rgba(255,204,0,.28);}
.result-covid{background:rgba(0,212,255,.07);border:1px solid rgba(0,212,255,.28);}
.cn{color:var(--green);}.cc{color:var(--red);}.cf{color:var(--yellow);}.cv{color:var(--cyan);}
.prob-wrap{margin-top:1.2rem;}
.prob-row{display:flex;align-items:center;gap:12px;margin-bottom:12px;}
.prob-name{font-size:.72rem;font-weight:500;width:76px;color:var(--muted);flex-shrink:0;}
.prob-track{flex:1;height:4px;background:rgba(255,255,255,.05);border-radius:2px;}
.prob-fill{height:4px;border-radius:2px;}
.prob-pct{font-size:.72rem;width:44px;text-align:right;color:var(--text);flex-shrink:0;}
.clinical-note{background:rgba(124,109,250,.07);border:1px solid rgba(124,109,250,.18);
               border-radius:10px;padding:1rem 1.2rem;font-size:.82rem;
               line-height:1.75;color:#a0aec0;margin-top:1.2rem;}
.clinical-note strong{color:var(--purple);}
.report-card{background:var(--bg-card2);border:1px solid var(--border);
             border-radius:12px;padding:1.4rem;margin-top:1.4rem;}
.report-card-title{font-family:'Syne',sans-serif;font-size:.65rem;font-weight:700;
                   letter-spacing:.18em;text-transform:uppercase;color:var(--muted);
                   margin-bottom:1rem;display:flex;align-items:center;gap:8px;}
.report-card-title::after{content:'';flex:1;height:1px;background:var(--border);}
.chat-bubble{border-radius:10px;padding:.9rem 1.1rem;margin-bottom:.9rem;font-size:.86rem;line-height:1.7;}
.bubble-user{background:rgba(0,212,255,.07);border-left:3px solid var(--cyan);}
.bubble-ai{background:rgba(0,232,135,.05);border-left:3px solid var(--green);}
.bubble-role{font-size:.6rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;margin-bottom:5px;}
.role-user{color:var(--cyan);}.role-ai{color:var(--green);}
.suggest-label{font-size:.7rem;color:var(--muted);margin-bottom:.7rem;}
.no-key-hint{text-align:center;padding:3rem 1rem;color:var(--muted);font-size:.85rem;line-height:2;}
.no-key-hint .icon{font-size:2.4rem;margin-bottom:.5rem;display:block;}
section[data-testid="stSidebar"]{background:#07090f!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] .block-container{padding:1.8rem 1.2rem!important;}
.sb-label{font-size:.6rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;
          color:var(--muted);margin-bottom:.7rem;margin-top:1.2rem;}
.sb-chip{background:var(--bg-card);border:1px solid var(--border);border-radius:8px;
         padding:.6rem .9rem;margin-bottom:.45rem;font-size:.78rem;color:var(--text);
         display:flex;justify-content:space-between;}
.sb-chip .sv{color:var(--cyan);font-weight:600;}
div[data-testid="stFileUploader"]{background:var(--bg-card2)!important;
    border:1px dashed rgba(0,212,255,.18)!important;border-radius:10px!important;}
.stButton>button{background:linear-gradient(135deg,#00d4ff,#00b3d4)!important;
    color:#050810!important;border:none!important;border-radius:8px!important;
    font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:.82rem!important;
    width:100%!important;padding:.7rem 1rem!important;transition:all .2s!important;}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 6px 24px rgba(0,212,255,.22)!important;}
div[data-testid="stChatInput"]>div{background:var(--bg-card2)!important;
    border:1px solid var(--border)!important;border-radius:10px!important;}
div[data-testid="stChatInput"] textarea{color:var(--text)!important;}
div[data-testid="stImage"] img{border-radius:10px;}
hr{border-color:var(--border)!important;margin:1.2rem 0!important;}
</style>
""", unsafe_allow_html=True)

# ══ Constants ═════════════════════════════════════════════════════════════════
CLASSES     = ["COVID-19","Chest Cancer","Fracture","Normal"]
BAR_COLOR   = {"Normal":"#00e887","Chest Cancer":"#ff3d57","Fracture":"#ffcc00","COVID-19":"#00d4ff"}
CSS_MAP     = {"Normal":"result-normal cn","Chest Cancer":"result-cancer cc",
               "Fracture":"result-fracture cf","COVID-19":"result-covid cv"}
ICON_MAP    = {"Normal":"✅","Chest Cancer":"⚠️","Fracture":"🦴","COVID-19":"🫁"}
NOTE_MAP    = {
    "Normal":       "No significant pathological findings detected. Lung fields are clear with no evidence of consolidation, effusion, or osseous abnormality. Routine follow-up as clinically indicated.",
    "Chest Cancer": "Potential malignant lesion identified. Immediate correlation with clinical history and tissue biopsy is recommended. Urgent oncology referral advised.",
    "Fracture":     "Osseous discontinuity pattern detected. Orthopedic consultation and additional orthogonal views (AP/lateral) are advised to confirm and characterize the fracture.",
    "COVID-19":     "Bilateral ground-glass opacities consistent with viral pneumonitis. RT-PCR confirmation and pulmonology review strongly recommended. Monitor oxygen saturation."
}
URGENCY_MAP = {"Normal":"Routine","Chest Cancer":"URGENT","Fracture":"Priority","COVID-19":"Priority"}
TRANSFORM   = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
GEMINI_KEY  = os.environ.get("GEMINI_API_KEY","")


# ══ Model ═════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else "cpu")
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048,256),
                         nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,len(CLASSES)))
    wp, loaded = "outputs/best_model.pt", False
    if os.path.exists(wp):
        m.load_state_dict(torch.load(wp, map_location=device))
        loaded = True
    m.to(device).eval()
    return m, device, loaded

def predict(model, device, img):
    t = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(t), dim=1)[0].cpu().tolist()
    return sorted([{"class":CLASSES[i],"pct":p*100} for i,p in enumerate(probs)],
                  key=lambda x: x["pct"], reverse=True)

def gemini_chat(key, messages, ctx=""):
    if not key:
        raise ValueError("GEMINI_API_KEY not set.")
    client = genai.Client(api_key=key)
    history = "".join(
        f"{'Radiologist' if m['role']=='user' else 'Assistant'}: {m['content']}\n"
        for m in messages[:-1]
    )
    system = ("You are RadVision AI — a concise, clinically precise radiology assistant. "
              "Interpret findings, suggest differentials, recommend next steps. "
              "Always advise physician review. Use **bold** for key terms. " + ctx)
    prompt = f"{system}\n\n{history}Radiologist: {messages[-1]['content']}\nAssistant:"
    r = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    if not r or not r.text:
        raise ValueError("Empty response from Gemini")
    return r.text

def get_ai_clinical_summary(key, patient_name, patient_age, patient_gender, results):
    if not key:
        return "AI clinical summary unavailable. Set GEMINI_API_KEY to enable."
    findings = ", ".join(f"{r['class']} ({r['pct']:.1f}%)" for r in results)
    top = results[0]
    prompt = (
        f"You are a senior radiologist writing a formal clinical impression for a radiology report. "
        f"Patient: {patient_name}, Age: {patient_age}, Gender: {patient_gender}. "
        f"CNN model probabilities: {findings}. Primary finding: {top['class']} at {top['pct']:.1f}% confidence.\n\n"
        f"Write exactly 3 paragraphs, each starting with its label in ALL CAPS followed by a colon:\n"
        f"CLINICAL IMPRESSION: (formal impression statement)\n"
        f"DIFFERENTIAL DIAGNOSIS: (2-3 differentials to consider)\n"
        f"RECOMMENDATIONS: (specific next steps for this patient)\n\n"
        f"Use formal clinical language. No markdown. No bullet points."
    )
    try:
        r = genai.Client(api_key=key).models.generate_content(
            model="gemini-2.5-flash", contents=prompt)
        return r.text.strip()
    except Exception as e:
        return f"AI summary generation failed: {str(e)}"


# ══ PDF Report — Clean White Professional ═════════════════════════════════════
def generate_pdf_report(patient_name, patient_age, patient_gender, patient_id,
                         referring_doctor, scan_type, clinical_notes,
                         results, ai_summary, orig_pil, chat_messages):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=15*mm, bottomMargin=15*mm)
    CW = A4[0] - 40*mm

    # ── Colour palette (clean white medical) ──────────────────────────────
    WHITE      = colors.white
    DARK_NAVY  = colors.HexColor("#0f3460")
    MED_BLUE   = colors.HexColor("#1e40af")
    ACCENT     = colors.HexColor("#0ea5e9")
    LIGHT_GREY = colors.HexColor("#f8fafc")
    MID_GREY   = colors.HexColor("#e2e8f0")
    TEXT_DARK  = colors.HexColor("#1e293b")
    TEXT_GREY  = colors.HexColor("#64748b")
    TEXT_LIGHT = colors.HexColor("#94a3b8")
    GREEN      = colors.HexColor("#16a34a")
    RED        = colors.HexColor("#dc2626")
    AMBER      = colors.HexColor("#d97706")
    PURPLE     = colors.HexColor("#7c3aed")
    GREEN_BG   = colors.HexColor("#f0fdf4")
    RED_BG     = colors.HexColor("#fef2f2")
    AMBER_BG   = colors.HexColor("#fffbeb")
    BLUE_BG    = colors.HexColor("#eff6ff")
    PURPLE_BG  = colors.HexColor("#faf5ff")

    RCOL    = {"Normal":GREEN,"Chest Cancer":RED,"Fracture":AMBER,"COVID-19":ACCENT}
    RCOL_BG = {"Normal":GREEN_BG,"Chest Cancer":RED_BG,"Fracture":AMBER_BG,"COVID-19":BLUE_BG}

    def ps(name, **kw):
        d = dict(fontName="Helvetica", fontSize=9, textColor=TEXT_DARK,
                 leading=14, backColor=WHITE)
        d.update(kw)
        return ParagraphStyle(name, **d)

    def sec(title, color=ACCENT):
        """Section header with left accent bar."""
        t = Table([[Paragraph(f"<b>{title}</b>",
                              ps(f"sh_{title}", fontName="Helvetica-Bold",
                                 fontSize=8, textColor=DARK_NAVY,
                                 backColor=LIGHT_GREY, letterSpacing=1))
                    ]], colWidths=[CW])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),LIGHT_GREY),
            ("LINEBEFORE",(0,0),(0,-1),4,color),
            ("LINEBELOW",(0,0),(-1,-1),0.5,MID_GREY),
            ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
            ("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),10),
        ]))
        return t

    def pil_rl(pil_img, w, h):
        b = io.BytesIO()
        pil_img.convert("RGB").save(b, "PNG")
        b.seek(0)
        return RLImage(b, width=w, height=h)

    now       = datetime.datetime.now()
    date_str  = now.strftime("%d %B %Y")
    time_str  = now.strftime("%H:%M")
    pid       = patient_id or f"RV-{now.strftime('%Y%m%d%H%M')}"
    report_id = f"RV-{now.strftime('%Y%m%d%H%M%S')}"
    story     = []
    top       = results[0]
    rc        = RCOL.get(top["class"], ACCENT)
    rc_bg     = RCOL_BG.get(top["class"], BLUE_BG)
    urgency   = URGENCY_MAP.get(top["class"], "Routine")
    urg_color = RED if urgency=="URGENT" else (AMBER if urgency=="Priority" else GREEN)

    # ════════════════════════════════════════════════════════════════
    # 1. LETTERHEAD
    # ════════════════════════════════════════════════════════════════
    hdr = Table([[
        Paragraph("<b>RadVision AI</b><br/>"
                  "<font size='8' color='#94a3b8'>AI-Powered Radiology Analysis</font>",
                  ps("logo", fontName="Helvetica-Bold", fontSize=20,
                     textColor=WHITE, backColor=DARK_NAVY, leading=28)),
        Paragraph(f"<font size='9' color='#cbd5e1'><b>RADIOLOGY REPORT</b></font><br/>"
                  f"<font size='7.5' color='#94a3b8'>Report ID: {report_id}</font><br/>"
                  f"<font size='7.5' color='#94a3b8'>Date: {date_str}  |  {time_str}</font>",
                  ps("meta", fontSize=8, alignment=TA_RIGHT,
                     textColor=WHITE, backColor=DARK_NAVY, leading=15)),
    ]], colWidths=[CW*0.58, CW*0.42])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),DARK_NAVY),
        ("TOPPADDING",(0,0),(-1,-1),16),("BOTTOMPADDING",(0,0),(-1,-1),16),
        ("LEFTPADDING",(0,0),(0,0),18),("RIGHTPADDING",(1,0),(1,0),18),
        ("LINEBELOW",(0,0),(-1,-1),3,ACCENT),
    ]))
    story += [hdr, Spacer(1,6)]

    # Disclaimer
    disc = Table([[Paragraph(
        "FOR RESEARCH USE ONLY — This AI-generated report is not a validated medical device "
        "and must be reviewed by a qualified radiologist before any clinical action.",
        ps("disc", fontSize=7.5, textColor=AMBER, backColor=AMBER_BG, alignment=TA_CENTER))
    ]], colWidths=[CW])
    disc.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),AMBER_BG),
        ("BOX",(0,0),(-1,-1),0.8,AMBER),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1),10),
    ]))
    story += [disc, Spacer(1,14)]

    # ════════════════════════════════════════════════════════════════
    # 2. PATIENT INFORMATION
    # ════════════════════════════════════════════════════════════════
    story += [sec("PATIENT INFORMATION"), Spacer(1,6)]

    rows = [
        # Row 1
        [ps("l1",fontSize=8,textColor=TEXT_GREY,backColor=LIGHT_GREY),
         "Patient Name", ps("v1",fontName="Helvetica-Bold",fontSize=9,textColor=TEXT_DARK,backColor=LIGHT_GREY),
         patient_name,
         ps("l2",fontSize=8,textColor=TEXT_GREY,backColor=LIGHT_GREY),
         "Patient ID",   ps("v2",fontName="Helvetica-Bold",fontSize=9,textColor=TEXT_DARK,backColor=LIGHT_GREY),
         pid],
        # Row 2
        [ps("l3",fontSize=8,textColor=TEXT_GREY,backColor=WHITE),
         "Age",          ps("v3",fontName="Helvetica-Bold",fontSize=9,textColor=TEXT_DARK,backColor=WHITE),
         f"{patient_age} years",
         ps("l4",fontSize=8,textColor=TEXT_GREY,backColor=WHITE),
         "Gender",       ps("v4",fontName="Helvetica-Bold",fontSize=9,textColor=TEXT_DARK,backColor=WHITE),
         patient_gender],
        # Row 3
        [ps("l5",fontSize=8,textColor=TEXT_GREY,backColor=LIGHT_GREY),
         "Referring Doctor", ps("v5",fontName="Helvetica-Bold",fontSize=9,textColor=TEXT_DARK,backColor=LIGHT_GREY),
         referring_doctor or "Not specified",
         ps("l6",fontSize=8,textColor=TEXT_GREY,backColor=LIGHT_GREY),
         "Scan Type",    ps("v6",fontName="Helvetica-Bold",fontSize=9,textColor=TEXT_DARK,backColor=LIGHT_GREY),
         scan_type],
    ]
    pt_data = []
    for sty_l, lbl, sty_v, val, sty_l2, lbl2, sty_v2, val2 in [
        (rows[0][0],rows[0][1],rows[0][2],rows[0][3],rows[0][4],rows[0][5],rows[0][6],rows[0][6] if len(rows[0])<8 else rows[0][6]),
    ]:
        pass

    # Simpler approach — build table data directly
    def ptrow(l1,v1,l2,v2,bg):
        return [
            Paragraph(l1, ps(f"pl{l1}",fontSize=8,textColor=TEXT_GREY,backColor=bg)),
            Paragraph(f"<b>{v1}</b>", ps(f"pv{l1}",fontSize=9,fontName="Helvetica-Bold",textColor=TEXT_DARK,backColor=bg)),
            Paragraph(l2, ps(f"pl{l2}",fontSize=8,textColor=TEXT_GREY,backColor=bg)),
            Paragraph(f"<b>{v2}</b>", ps(f"pv{l2}",fontSize=9,fontName="Helvetica-Bold",textColor=TEXT_DARK,backColor=bg)),
        ]

    pt_tbl = Table([
        ptrow("Patient Name", patient_name,          "Patient ID",       pid,                       LIGHT_GREY),
        ptrow("Age",          f"{patient_age} yrs",  "Gender",           patient_gender,             WHITE),
        ptrow("Referring Dr", referring_doctor or "N/A", "Scan Type",    scan_type,                 LIGHT_GREY),
        ptrow("Report Date",  date_str,               "Report Time",      time_str,                  WHITE),
    ], colWidths=[CW*0.18, CW*0.32, CW*0.18, CW*0.32])
    pt_tbl.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.4,MID_GREY),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(-1,-1),10),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[LIGHT_GREY,WHITE]),
    ]))
    story += [pt_tbl, Spacer(1,14)]

    # ════════════════════════════════════════════════════════════════
    # 3. PRIMARY FINDING BANNER
    # ════════════════════════════════════════════════════════════════
    story += [sec("PRIMARY FINDING", rc), Spacer(1,6)]

    banner = Table([[
        Paragraph(f"<b>{top['class'].upper()}</b><br/>"
                  f"<font size='9' color='#64748b'>Primary AI Detection</font>",
                  ps("bf", fontName="Helvetica-Bold", fontSize=22,
                     textColor=rc, backColor=rc_bg, leading=30)),
        Paragraph(f"<b>{top['pct']:.1f}%</b><br/>"
                  f"<font size='8' color='#64748b'>AI Confidence</font>",
                  ps("bc", fontName="Helvetica-Bold", fontSize=20,
                     textColor=rc, backColor=rc_bg, alignment=TA_CENTER, leading=28)),
        Paragraph(f"<b>{urgency}</b><br/>"
                  f"<font size='8' color='#64748b'>Priority Level</font>",
                  ps("bu", fontName="Helvetica-Bold", fontSize=18,
                     textColor=urg_color, backColor=rc_bg, alignment=TA_RIGHT, leading=26)),
    ]], colWidths=[CW*0.50, CW*0.25, CW*0.25])
    banner.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),rc_bg),
        ("BOX",(0,0),(-1,-1),2,rc),
        ("LINEAFTER",(0,0),(1,0),0.5,MID_GREY),
        ("TOPPADDING",(0,0),(-1,-1),16),("BOTTOMPADDING",(0,0),(-1,-1),16),
        ("LEFTPADDING",(0,0),(0,0),18),("RIGHTPADDING",(-1,0),(-1,0),18),
        ("LEFTPADDING",(1,0),(2,0),12),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story += [banner, Spacer(1,14)]

    # ════════════════════════════════════════════════════════════════
    # 4. CONFIDENCE BREAKDOWN
    # ════════════════════════════════════════════════════════════════
    story += [sec("AI CONFIDENCE BREAKDOWN"), Spacer(1,6)]

    indic = {
        "Normal":       "No significant pathology detected",
        "Chest Cancer": "Malignancy pattern — urgent review required",
        "Fracture":     "Osseous discontinuity pattern identified",
        "COVID-19":     "Bilateral ground-glass opacities pattern"
    }
    hdr_row = [
        Paragraph("<b>Condition</b>",           ps("th0",fontName="Helvetica-Bold",fontSize=8.5,textColor=WHITE,backColor=DARK_NAVY)),
        Paragraph("<b>Score</b>",               ps("th1",fontName="Helvetica-Bold",fontSize=8.5,textColor=WHITE,backColor=DARK_NAVY,alignment=TA_CENTER)),
        Paragraph("<b>Clinical Indication</b>", ps("th2",fontName="Helvetica-Bold",fontSize=8.5,textColor=WHITE,backColor=DARK_NAVY)),
        Paragraph("<b>Status</b>",              ps("th3",fontName="Helvetica-Bold",fontSize=8.5,textColor=WHITE,backColor=DARK_NAVY,alignment=TA_CENTER)),
    ]
    tbl_data = [hdr_row]
    for i, r in enumerate(results):
        is_top = (r == top)
        bg2 = rc_bg if is_top else (LIGHT_GREY if i%2==0 else WHITE)
        rc2 = RCOL.get(r["class"], ACCENT)
        tbl_data.append([
            Paragraph(f"{'<b>' if is_top else ''}{r['class']}{'</b>' if is_top else ''}",
                      ps(f"td0{i}", fontSize=9,
                         fontName="Helvetica-Bold" if is_top else "Helvetica",
                         textColor=rc2 if is_top else TEXT_DARK, backColor=bg2)),
            Paragraph(f"{'<b>' if is_top else ''}{r['pct']:.2f}%{'</b>' if is_top else ''}",
                      ps(f"td1{i}", fontSize=9, alignment=TA_CENTER,
                         fontName="Helvetica-Bold" if is_top else "Helvetica",
                         textColor=rc2 if is_top else TEXT_DARK, backColor=bg2)),
            Paragraph(indic.get(r["class"],"—"),
                      ps(f"td2{i}", fontSize=8.5, textColor=TEXT_GREY, backColor=bg2)),
            Paragraph("PRIMARY" if is_top else "—",
                      ps(f"td3{i}", fontSize=8,
                         fontName="Helvetica-Bold" if is_top else "Helvetica",
                         textColor=rc2 if is_top else TEXT_GREY,
                         alignment=TA_CENTER, backColor=bg2)),
        ])
    conf = Table(tbl_data, colWidths=[CW*0.22, CW*0.13, CW*0.46, CW*0.19])
    conf.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),DARK_NAVY),
        ("GRID",(0,0),(-1,-1),0.4,MID_GREY),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(-1,-1),10),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story += [conf, Spacer(1,14)]

    # ════════════════════════════════════════════════════════════════
    # 5. SCAN IMAGE + DETAILS
    # ════════════════════════════════════════════════════════════════
    story += [sec("SCAN IMAGE & DETAILS"), Spacer(1,6)]

    img_w   = CW * 0.46
    meta_w  = CW - img_w - 5*mm
    orig_rl = pil_rl(orig_pil.resize((380,380)), img_w, img_w)

    scan_info = [
        ("Scan Type",       scan_type),
        ("Analysis Date",   date_str),
        ("AI Model",        "ResNet-50 CNN"),
        ("Val Accuracy",    "97.88%"),
        ("Input Size",      "224 x 224 px"),
        ("Primary Finding", top["class"]),
        ("Confidence",      f"{top['pct']:.1f}%"),
        ("Priority",        urgency),
    ]
    meta_rows = []
    for i,(lbl,val) in enumerate(scan_info):
        bg3 = LIGHT_GREY if i%2==0 else WHITE
        is_pf = lbl in ("Primary Finding","Confidence")
        meta_rows.append([
            Paragraph(lbl, ps(f"ml{i}",fontSize=8,textColor=TEXT_GREY,backColor=bg3)),
            Paragraph(f"<b>{val}</b>",
                      ps(f"mv{i}",fontSize=8.5,fontName="Helvetica-Bold",
                         textColor=rc if is_pf else TEXT_DARK, backColor=bg3)),
        ])
    meta_tbl = Table(meta_rows, colWidths=[meta_w*0.48, meta_w*0.52])
    meta_tbl.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.3,MID_GREY),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1),8),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[LIGHT_GREY,WHITE]),
    ]))
    img_row = Table([[orig_rl, meta_tbl]], colWidths=[img_w+4*mm, meta_w])
    img_row.setStyle(TableStyle([
        ("ALIGN",(0,0),(0,0),"CENTER"),("VALIGN",(0,0),(-1,-1),"TOP"),
        ("BOX",(0,0),(-1,-1),0.5,MID_GREY),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
    ]))
    story += [img_row, Spacer(1,14)]

    # ════════════════════════════════════════════════════════════════
    # 6. AI CLINICAL NOTE
    # ════════════════════════════════════════════════════════════════
    story += [sec("AI CLINICAL NOTE", ACCENT), Spacer(1,6)]
    note = Table([[Paragraph(NOTE_MAP[top["class"]],
                             ps("note",fontSize=9.5,textColor=TEXT_DARK,
                                backColor=BLUE_BG,leading=16))
                   ]], colWidths=[CW])
    note.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),BLUE_BG),
        ("BOX",(0,0),(-1,-1),1,ACCENT),
        ("LINEBEFORE",(0,0),(0,-1),4,ACCENT),
        ("TOPPADDING",(0,0),(-1,-1),12),("BOTTOMPADDING",(0,0),(-1,-1),12),
        ("LEFTPADDING",(0,0),(-1,-1),14),("RIGHTPADDING",(0,0),(-1,-1),14),
    ]))
    story += [note, Spacer(1,14)]

    # ════════════════════════════════════════════════════════════════
    # 7. GEMINI AI CLINICAL IMPRESSION
    # ════════════════════════════════════════════════════════════════
    story += [sec("GEMINI AI CLINICAL IMPRESSION", PURPLE), Spacer(1,0)]

    # AI badge bar
    badge = Table([[Paragraph(
        "AI GENERATED  |  Gemini AI  |  For Clinical Review Only — Not a substitute for radiologist review",
        ps("badge",fontSize=7.5,fontName="Helvetica-Bold",textColor=WHITE,
           backColor=PURPLE,alignment=TA_CENTER))
    ]], colWidths=[CW])
    badge.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),PURPLE),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))
    story += [badge, Spacer(1,4)]

    # Parse the 3 labelled paragraphs from Gemini
    SECTION_LABELS = {
        "CLINICAL IMPRESSION:":  (DARK_NAVY,  BLUE_BG),
        "DIFFERENTIAL DIAGNOSIS:":(PURPLE,    PURPLE_BG),
        "RECOMMENDATIONS:":      (GREEN,      GREEN_BG),
    }
    clean = ai_summary.replace("**","").replace("*","")
    # Split on label keywords
    import re
    parts = re.split(r'(CLINICAL IMPRESSION:|DIFFERENTIAL DIAGNOSIS:|RECOMMENDATIONS:)', clean)
    current_label, current_text = None, []
    parsed = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in SECTION_LABELS:
            if current_label and current_text:
                parsed.append((current_label, " ".join(current_text)))
            current_label = part
            current_text  = []
        else:
            current_text.append(part)
    if current_label and current_text:
        parsed.append((current_label, " ".join(current_text)))

    if parsed:
        for lbl, body in parsed:
            lc, lb_bg = SECTION_LABELS.get(lbl, (DARK_NAVY, BLUE_BG))
            lbl_clean = lbl.rstrip(":")
            ai_row = Table([[
                Paragraph(f"<b>{lbl_clean}</b>",
                          ps(f"al_{lbl_clean}", fontSize=8, fontName="Helvetica-Bold",
                             textColor=WHITE, backColor=lc, alignment=TA_CENTER)),
                Paragraph(body, ps(f"ab_{lbl_clean}", fontSize=9.5, textColor=TEXT_DARK,
                                   backColor=lb_bg, leading=15)),
            ]], colWidths=[CW*0.22, CW*0.78])
            ai_row.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(0,0),lc),
                ("BACKGROUND",(1,0),(1,0),lb_bg),
                ("GRID",(0,0),(-1,-1),0.4,MID_GREY),
                ("VALIGN",(0,0),(-1,-1),"TOP"),
                ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
                ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
            ]))
            story += [ai_row, Spacer(1,3)]
    else:
        # Fallback: show all as one block
        fb = Table([[Paragraph(clean, ps("fb",fontSize=9.5,textColor=TEXT_DARK,
                                          backColor=PURPLE_BG,leading=15))
                     ]], colWidths=[CW])
        fb.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),PURPLE_BG),
            ("BOX",(0,0),(-1,-1),0.5,PURPLE),
            ("TOPPADDING",(0,0),(-1,-1),12),("BOTTOMPADDING",(0,0),(-1,-1),12),
            ("LEFTPADDING",(0,0),(-1,-1),12),
        ]))
        story.append(fb)
    story.append(Spacer(1,14))

    # ════════════════════════════════════════════════════════════════
    # 8. DOCTOR'S CLINICAL NOTES (optional)
    # ════════════════════════════════════════════════════════════════
    if clinical_notes and clinical_notes.strip():
        story += [sec("REFERRING DOCTOR'S NOTES", colors.HexColor("#7c3aed")), Spacer(1,6)]
        dn = Table([[Paragraph(clinical_notes,
                               ps("dn",fontSize=9.5,textColor=TEXT_DARK,
                                  backColor=PURPLE_BG,leading=15))
                     ]], colWidths=[CW])
        dn.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),PURPLE_BG),
            ("BOX",(0,0),(-1,-1),0.5,MID_GREY),
            ("LINEBEFORE",(0,0),(0,-1),4,PURPLE),
            ("TOPPADDING",(0,0),(-1,-1),12),("BOTTOMPADDING",(0,0),(-1,-1),12),
            ("LEFTPADDING",(0,0),(-1,-1),14),("RIGHTPADDING",(0,0),(-1,-1),14),
        ]))
        story += [dn, Spacer(1,14)]

    # ════════════════════════════════════════════════════════════════
    # 9. AI CHAT TRANSCRIPT (optional)
    # ════════════════════════════════════════════════════════════════
    msgs = [m for m in chat_messages if m.get("content","").strip()][-6:]
    if msgs:
        story += [sec("AI DISCUSSION TRANSCRIPT", ACCENT), Spacer(1,6)]
        for msg in msgs:
            is_user = msg["role"] == "user"
            role_lbl = "RADIOLOGIST" if is_user else "RADVISION AI"
            bg4 = BLUE_BG if is_user else GREEN_BG
            bc2 = ACCENT  if is_user else GREEN
            content = msg["content"].replace("**","")[:500]
            mt = Table([[Paragraph(
                f"<font size='7.5' color='{'#0369a1' if is_user else '#15803d'}'><b>{role_lbl}</b></font>"
                f"<br/><font size='9'>{content}</font>",
                ps(f"cm_{role_lbl}", backColor=bg4, leading=14))
            ]], colWidths=[CW])
            mt.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,-1),bg4),
                ("LINEBEFORE",(0,0),(0,-1),3,bc2),
                ("BOX",(0,0),(-1,-1),0.3,MID_GREY),
                ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
                ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
            ]))
            story += [mt, Spacer(1,4)]
        story.append(Spacer(1,8))

    # ════════════════════════════════════════════════════════════════
    # 10. FOOTER
    # ════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=1, color=MID_GREY, spaceBefore=6))
    story.append(Spacer(1,6))
    foot = Table([[
        Paragraph(f"<b>RadVision AI</b>  |  ResNet-50 CNN  |  97.88% Validation Accuracy<br/>"
                  f"<font size='7' color='#94a3b8'>Patient: {patient_name}  |  Report: {report_id}</font>",
                  ps("fl",fontSize=8,textColor=TEXT_GREY,leading=13)),
        Paragraph(f"<font color='#94a3b8'>Generated: {date_str} at {time_str}</font><br/>"
                  f"<font size='7' color='#94a3b8'>Requires radiologist validation before clinical use.</font>",
                  ps("fr",fontSize=8,textColor=TEXT_GREY,alignment=TA_RIGHT,leading=13)),
    ]], colWidths=[CW*0.6, CW*0.4])
    foot.setStyle(TableStyle([
        ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2),
    ]))
    story.append(foot)

    doc.build(story)
    buf.seek(0)
    return buf


# ══ Boot ══════════════════════════════════════════════════════════════════════
model, device, weights_loaded = load_model()

# ══ Top Bar ═══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="topbar">
  <div><span class="logo">RadVision AI</span>
       <span class="logo-sub">Radiology Analysis Platform</span></div>
  <div style="display:flex;align-items:center;gap:14px;">
    <span class="badge-live">SYSTEM ACTIVE</span>
    <span style="font-size:.7rem;color:var(--muted);">
      ResNet-50 &nbsp;·&nbsp; {"Trained" if weights_loaded else "Base"} Weights &nbsp;·&nbsp; {device.upper()}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══ Sidebar ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-label">Model Status</div>', unsafe_allow_html=True)
    if weights_loaded:
        st.success("✓ Trained weights loaded  (97.88% val acc)")
    else:
        st.warning("⚠ Base weights — run trainer.py first")
    st.markdown('<div class="sb-label">System Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="sb-chip"><span>Architecture</span><span class="sv">ResNet-50</span></div>
<div class="sb-chip"><span>Device</span><span class="sv">{device.upper()}</span></div>
<div class="sb-chip"><span>Input Size</span><span class="sv">224 × 224</span></div>
<div class="sb-chip"><span>Pathologies</span><span class="sv">4 Classes</span></div>
<div class="sb-chip"><span>Training Images</span><span class="sv">5,925</span></div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="font-size:.65rem;color:var(--muted);line-height:1.9;">'
                '⚠ For research use only.<br>Not a validated medical device.<br>'
                'Always consult a qualified radiologist.</div>', unsafe_allow_html=True)

# ══ Main Layout ═══════════════════════════════════════════════════════════════
col_l, col_r = st.columns([1,1], gap="large")

# ── LEFT ──────────────────────────────────────────────────────────────────────
with col_l:
    st.markdown('<div class="panel-title">🔬 Scan Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="stats-row">
  <div class="stat-chip"><div class="sv">97.9%</div><div class="sl">Val Accuracy</div></div>
  <div class="stat-chip"><div class="sv">5,925</div><div class="sl">Training Images</div></div>
  <div class="stat-chip"><div class="sv">4</div><div class="sl">Pathologies</div></div>
</div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop a radiology scan", type=["png","jpg","jpeg"],
                                label_visibility="visible")
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption=uploaded.name, use_column_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡  Run AI Detection"):
            with st.spinner("Analyzing with ResNet-50…"):
                st.session_state["results"]  = predict(model, device, img)
                st.session_state["orig_img"] = img
    else:
        st.markdown("""
<div class="upload-hint">
  <span class="icon">🩻</span>
  Upload a chest X-ray or radiology scan<br>
  <small style="font-size:.78rem;">Supports PNG · JPG · JPEG</small>
</div>""", unsafe_allow_html=True)

    if "results" in st.session_state:
        res = st.session_state["results"]
        top = res[0]
        css = CSS_MAP[top["class"]]
        st.markdown(f"""
<div class="detection-result {css.split()[0]}">
  <span class="d-icon">{ICON_MAP[top['class']]}</span>
  <div>
    <div class="d-label {css.split()[1]}">{top['class'].upper()}</div>
    <div class="d-sub">Primary Detection · {top['pct']:.1f}% confidence</div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="prob-wrap">', unsafe_allow_html=True)
        for r in res:
            c = BAR_COLOR[r["class"]]
            st.markdown(f"""
<div class="prob-row">
  <span class="prob-name">{r['class']}</span>
  <div class="prob-track"><div class="prob-fill" style="width:{r['pct']:.1f}%;background:{c};"></div></div>
  <span class="prob-pct">{r['pct']:.1f}%</span>
</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
<div class="clinical-note">
  <strong>AI Clinical Note</strong> — {NOTE_MAP[top['class']]}
</div>""", unsafe_allow_html=True)

        # ── Patient Info + Report ─────────────────────────────────────────
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown('<div class="report-card-title">📋 Patient Information & Report</div>',
                    unsafe_allow_html=True)

        pc1, pc2 = st.columns(2)
        with pc1:
            patient_name = st.text_input("Patient Name *", placeholder="e.g. John Smith", key="pt_name")
            patient_age  = st.number_input("Age *", min_value=1, max_value=120, value=35, key="pt_age")
            patient_id   = st.text_input("Patient ID", placeholder="e.g. PT-00123", key="pt_id")
        with pc2:
            patient_gender   = st.selectbox("Gender *", ["Male","Female","Other"], key="pt_gender")
            scan_type        = st.selectbox("Scan Type",
                                             ["Chest X-Ray","CT Scan","MRI","PET Scan","Ultrasound"],
                                             key="pt_scan")
            referring_doctor = st.text_input("Referring Doctor", placeholder="Dr. Name", key="pt_doc")

        clinical_notes = st.text_area("Doctor's Clinical Notes (optional)",
                                       placeholder="Enter any additional clinical observations, symptoms or history…",
                                       height=80, key="pt_notes")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("📄  Generate Clinical PDF Report"):
            if not patient_name.strip():
                st.error("Please enter the patient name to generate the report.")
            else:
                with st.spinner("Generating Gemini AI clinical impression…"):
                    ai_summary = get_ai_clinical_summary(
                        GEMINI_KEY, patient_name, patient_age,
                        patient_gender, st.session_state["results"]
                    )
                with st.spinner("Building professional PDF report…"):
                    pdf_buf = generate_pdf_report(
                        patient_name    = patient_name,
                        patient_age     = patient_age,
                        patient_gender  = patient_gender,
                        patient_id      = patient_id,
                        referring_doctor= referring_doctor,
                        scan_type       = scan_type,
                        clinical_notes  = clinical_notes,
                        results         = st.session_state["results"],
                        ai_summary      = ai_summary,
                        orig_pil        = st.session_state.get("orig_img", img),
                        chat_messages   = st.session_state.get("messages", [])
                    )
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"RadVision_{patient_name.replace(' ','_')}_{ts}.pdf"
                st.download_button(
                    label="⬇  Download PDF Report",
                    data=pdf_buf, file_name=fname,
                    mime="application/pdf", use_container_width=True
                )
                st.success("✓ Report ready — click above to download.")

        st.markdown('</div>', unsafe_allow_html=True)

# ── RIGHT ─────────────────────────────────────────────────────────────────────
with col_r:
    dot = "🟢" if GEMINI_KEY else "🔴"
    st.markdown(f'<div class="panel-title">🤖 Clinical AI Discussion &nbsp;{dot}</div>',
                unsafe_allow_html=True)

    if not GEMINI_KEY:
        st.markdown("""
<div class="no-key-hint">
  <span class="icon">💬</span>
  Gemini AI is not configured.<br>
  <small style="font-size:.78rem;">
    Set <code>GEMINI_API_KEY</code> environment variable<br>
    and restart the app to enable clinical discussion.
  </small>
</div>""", unsafe_allow_html=True)
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            cls  = "bubble-user" if msg["role"]=="user" else "bubble-ai"
            role = "You"         if msg["role"]=="user" else "RadVision AI"
            rcls = "role-user"   if msg["role"]=="user" else "role-ai"
            st.markdown(f'<div class="chat-bubble {cls}"><div class="bubble-role {rcls}">'
                        f'{role}</div>{msg["content"]}</div>', unsafe_allow_html=True)

        if "results" in st.session_state:
            top2 = st.session_state["results"][0]
            st.markdown('<div class="suggest-label">💡 Suggested Questions:</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)
            suggestions = [
                (c1, f"What does {top2['class']} mean?",    f"What does {top2['class']} mean?"),
                (c2, "Recommended next steps?",              "What are the recommended next steps?"),
                (c3, "Treatment options?",                   "What are the available treatment options?"),
                (c4, "Risk factors?",                        "What are the risk factors for this condition?"),
            ]
            for col, label, prompt_text in suggestions:
                with col:
                    if st.button(label, key=f"sq_{label}", use_container_width=True):
                        st.session_state.messages.append({"role":"user","content":prompt_text})
                        _f = ", ".join(f"{x['class']} {x['pct']:.1f}%" for x in st.session_state["results"])
                        ctx = f"Scan results: {_f}. Primary: {top2['class']}. "
                        with st.spinner("Thinking…"):
                            try:
                                reply = gemini_chat(GEMINI_KEY, st.session_state.messages, ctx)
                                st.session_state.messages.append({"role":"assistant","content":reply})
                                st.rerun()
                            except Exception as e:
                                st.session_state.messages.pop()
                                st.error(str(e))
            st.markdown("<br>", unsafe_allow_html=True)

        if prompt := st.chat_input("Ask about findings, treatment, prognosis…"):
            st.session_state.messages.append({"role":"user","content":prompt})
            ctx = ""
            if "results" in st.session_state:
                _f2  = ", ".join(f"{x['class']} {x['pct']:.1f}%" for x in st.session_state["results"])
                top3 = st.session_state["results"][0]
                ctx  = f"Scan results: {_f2}. Primary: {top3['class']} at {top3['pct']:.1f}% confidence. "
            with st.spinner("Thinking…"):
                try:
                    reply = gemini_chat(GEMINI_KEY, st.session_state.messages, ctx)
                    st.session_state.messages.append({"role":"assistant","content":reply})
                    st.rerun()
                except ValueError as ve:
                    st.session_state.messages.pop()
                    st.error(str(ve))
                except Exception as e:
                    st.session_state.messages.pop()
                    st.error(f"❌ Unexpected error: {str(e)}")

        if st.session_state.get("messages"):
            if st.button("🗑  Clear conversation"):
                st.session_state.messages = []
                st.rerun()
