"""TransferLens visual theme: a scout's tactics board.

Pitch-green ground, chalk lines and text, one amber accent (the captain's armband).
Archivo is used two ways: expanded and heavy for headings and stat numbers, normal width for reading.
"""
import streamlit as st

PITCH, TURF, LINE, CHALK, FOG, AMBER = "#0F2E22", "#153A2C", "#2C5A47", "#EDEFE6", "#A9B8AE", "#F2B134"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..900&display=swap');

.stApp { background: %(PITCH)s; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stMarkdown, [data-testid="stCaptionContainer"] {
    font-family: 'Archivo', system-ui, sans-serif;
}
.block-container { padding-top: 1.5rem; max-width: 1240px; }

/* headings and numbers: expanded, heavy, like shirt numbers */
.stApp h1, .stApp h2, .stApp h3, .tl-title, [data-testid="stMetricValue"] {
    font-family: 'Archivo', system-ui, sans-serif;
    font-stretch: 125%%;
    letter-spacing: -0.01em;
    color: %(CHALK)s;
}
.stApp h2, .stApp h3 { font-weight: 800; }
.stApp h3 { font-size: 1.35rem; }

/* hero: the halfway line */
.tl-hero {
    position: relative; overflow: hidden;
    padding: 2.6rem 0 2.2rem 0; margin-bottom: 0.6rem;
    border-bottom: 2px solid rgba(237, 239, 230, 0.16);
}
.tl-hero::before {               /* centre circle */
    content: ""; position: absolute; width: 360px; height: 360px; border-radius: 50%%;
    border: 2px solid rgba(237, 239, 230, 0.13);
    right: 8%%; top: 50%%; transform: translate(50%%, -50%%);
}
.tl-hero::after {                /* halfway line */
    content: ""; position: absolute; top: 0; bottom: 0; width: 2px;
    right: 8%%; background: rgba(237, 239, 230, 0.13);
}
.tl-spot {                       /* centre spot, the one accent */
    position: absolute; width: 12px; height: 12px; border-radius: 50%%; background: %(AMBER)s;
    right: 8%%; top: 50%%; transform: translate(50%%, -50%%); z-index: 1;
}
.tl-title {
    position: relative; z-index: 1; margin: 0;
    font-size: clamp(2.4rem, 5.5vw, 4.4rem); font-weight: 900; line-height: 0.95;
}
.tl-sub {
    position: relative; z-index: 1; max-width: 62ch; margin: 1rem 0 0 0;
    color: %(FOG)s; font-size: 1.05rem; line-height: 1.55;
}

/* tabs: chalk text, amber marker on the active one */
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid %(LINE)s; }
.stTabs [data-baseweb="tab"] { padding: 0.7rem 0; color: %(FOG)s; font-weight: 600; }
.stTabs [data-baseweb="tab"] p { font-size: 1rem; }
.stTabs [aria-selected="true"] { color: %(CHALK)s; }
.stTabs [data-baseweb="tab-highlight"] { background-color: %(AMBER)s; height: 3px; }

/* stats: programme-style lines, not boxes */
[data-testid="stMetric"] { border-top: 2px solid rgba(237, 239, 230, 0.22); padding-top: 0.7rem; }
[data-testid="stMetricLabel"] p { color: %(FOG)s; font-size: 0.92rem; }
[data-testid="stMetricValue"] { font-weight: 800; font-size: 2rem; }

/* sidebar */
[data-testid="stSidebar"] { background: %(TURF)s; border-right: 1px solid %(LINE)s; }
.tl-mark { display: flex; align-items: center; gap: 0.7rem; margin: 0.2rem 0 1rem 0; }
.tl-mark span { font-family: 'Archivo', sans-serif; font-stretch: 125%%; font-weight: 900; font-size: 1.35rem; color: %(CHALK)s; }

/* panels, expanders, chat */
[data-testid="stExpander"] details { border-color: %(LINE)s; background: %(TURF)s; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: %(LINE)s; }
[data-testid="stChatMessage"] { background: %(TURF)s; border: 1px solid %(LINE)s; }
[data-testid="stForm"] { border-color: %(LINE)s; background: rgba(21, 58, 44, 0.55); }

/* primary buttons in amber with pitch-green text */
.stApp button[kind="primary"], .stApp [data-testid="stBaseButton-primary"] {
    background: %(AMBER)s; border-color: %(AMBER)s; color: %(PITCH)s; font-weight: 700;
}
.stApp button[kind="primary"]:hover, .stApp [data-testid="stBaseButton-primary"]:hover {
    background: #FFC85A; border-color: #FFC85A; color: %(PITCH)s;
}
.stApp button:focus-visible { outline: 2px solid %(AMBER)s; outline-offset: 2px; }

@media (max-width: 640px) {
    .tl-hero::before { width: 220px; height: 220px; right: 0; }
    .tl-hero::after, .tl-spot { right: 0; }
}
</style>
""" % dict(PITCH=PITCH, TURF=TURF, LINE=LINE, CHALK=CHALK, FOG=FOG, AMBER=AMBER)

MARK = """
<div class="tl-mark">
  <svg width="30" height="30" viewBox="0 0 30 30" aria-hidden="true">
    <circle cx="15" cy="15" r="12" fill="none" stroke="%s" stroke-width="2"/>
    <line x1="15" y1="1" x2="15" y2="29" stroke="%s" stroke-width="2"/>
    <circle cx="15" cy="15" r="3" fill="%s"/>
  </svg>
  <span>TransferLens</span>
</div>
""" % (CHALK, CHALK, AMBER)


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(n_players):
    st.markdown(
        '<div class="tl-hero"><div class="tl-spot"></div>'
        '<h1 class="tl-title">TransferLens</h1>'
        '<p class="tl-sub">Recruitment decisions you can check. Fair values, move probabilities and signing '
        'plans for %s players in the covered leagues, each number from a model tested on seasons it never saw.</p>'
        '</div>' % format(n_players, ","), unsafe_allow_html=True)
