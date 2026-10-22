"""Tomato Leaf Check: Streamlit app for the Group 23 mini project
"AI-Based Early Detection and Management System for Crop Diseases and Pest Infestations".

Run locally:   streamlit run app.py
Needs, next to this file:
  models/final_model.keras, models/final_model_info.json   (from Drive: TomatoAI/models)
  action_cards.json                                          (advice for each class)
  samples/*.jpg and samples/thumbs/*.jpg                     (example leaves)
"""
import io
import os
import json
import time
import base64
from html import escape

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from matplotlib import colormaps
from PIL import Image, ImageOps

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(APP_DIR, 'models'))
SAMPLE_DIR = os.path.join(APP_DIR, 'samples')
FONTS = ('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT@9..144,400..700,50'
         '&family=Atkinson+Hyperlegible:wght@400;700&display=swap')

# what each class is, for colour coding: (group, colour)
GROUP = {'Early_blight': ('Disease', '#B8392A'), 'Late_blight': ('Disease', '#B8392A'),
         'Spotted Wilt Virus': ('Virus', '#7B3F94'), 'Leaf Miner': ('Pest', '#B26A00'),
         'Magnesium Deficiency': ('Nutrient deficiency', '#6E6A00'), 'Nitrogen Deficiency': ('Nutrient deficiency', '#6E6A00'),
         'Healthy': ('Healthy', '#2E7D3A')}
EXAMPLE = {'Early_blight': 'early_blight.jpg', 'Late_blight': 'late_blight.jpg', 'Spotted Wilt Virus': 'spotted_wilt.jpg',
           'Leaf Miner': 'leaf_miner.jpg', 'Magnesium Deficiency': 'magnesium_deficiency.jpg',
           'Nitrogen Deficiency': 'nitrogen_deficiency.jpg', 'Healthy': 'healthy.jpg'}
HARD_CASE = 'field_photo_late_blight.jpg'
URGENCY = {'high': 'Act within days', 'medium': 'Act this week', 'low': 'Not urgent: fix at the next feeding',
           'none': 'Nothing to do'}
MODEL_TEXT = {
    'mobilenetv2_probe': 'frozen MobileNetV2 with a logistic-regression classifier, trained on lab photos of single leaves',
    'mobilenetv2_probe_field': 'frozen MobileNetV2 with a logistic-regression classifier, trained on 1,778 lab photos and 1,141 leaves cut from field photos',
    'mobilenetv2_all': 'MobileNetV2 fine-tuned on lab photos of single leaves',
}

st.set_page_config(page_title='Tomato Leaf Check', page_icon='🍅', layout='wide', initial_sidebar_state='collapsed')

CSS = """
<style>
@import url('__FONTS__');
@property --pa { syntax: '<number>'; inherits: false; initial-value: 0; }
:root {
  --forest: #0D2A1C; --canopy: #17472F; --emerald: #1F7A4A; --leaf: #5BA44A; --mint: #BCD8C3; --paper: #F3F6F0; --sheet: #FFFFFF;
  --ink: #13251B; --muted: #536459; --rule: #DCE4D8; --tomato: #E0452F; --tomato-text: #B3301F; --sun: #F0AE33;
  --sun-text: #855508; --sun-soft: #FDF3DE; --leaf-text: #2B6B27;
  --display: 'Fraunces', Georgia, serif; --body: 'Atkinson Hyperlegible', system-ui, sans-serif;
}
[data-testid="stAppViewContainer"], [data-testid="stApp"] { background: var(--paper); color: var(--ink); }
[data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"], .stButton button, [data-testid="stPopover"] button { font-family: var(--body); }
[data-testid="stMainBlockContainer"], .block-container { max-width: 1640px; padding: 1.2rem 3rem 3rem; }
@media (max-width: 760px) { [data-testid="stMainBlockContainer"], .block-container { padding: 0.8rem 1rem 2rem; } }
header[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stToolbar"], #MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
h1, h2, h3, h4 { font-family: var(--display) !important; color: var(--ink); }

/* top bar */
.bar-top { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin: 0 0 14px; }
.brand { display: flex; align-items: center; gap: 10px; font-family: var(--display); font-size: 1.35rem; font-weight: 600; color: var(--ink); }
.brand svg { width: 30px; height: 30px; }
.bar-top .who { color: var(--muted); font-size: 0.95rem; text-align: right; }
@media (max-width: 640px) { .bar-top { flex-direction: column; align-items: flex-start; gap: 4px; } .bar-top .who { text-align: left; font-size: 0.85rem; } }

/* top row: hero card + upload card */
.st-key-top [data-testid="stHorizontalBlock"] { align-items: stretch; }
.st-key-top [data-testid="stColumn"]:first-child > div,
.st-key-top [data-testid="stColumn"]:first-child [data-testid="stVerticalBlock"],
.st-key-top [data-testid="stColumn"]:first-child [data-testid="stElementContainer"],
.st-key-top [data-testid="stColumn"]:first-child [data-testid="stMarkdown"],
.st-key-top [data-testid="stColumn"]:first-child [data-testid="stMarkdownContainer"],
.st-key-top [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"],
.st-key-top [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] { height: 100%; }
.st-key-upload_card [data-testid="stLayoutWrapper"] { width: 100%; }
.st-key-top [data-testid="stColumn"]:first-child [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
.hero { position: relative; overflow: hidden; box-sizing: border-box; height: 100%; min-height: 400px;
        background: radial-gradient(110% 130% at 100% 0%, #2A6B45 0%, var(--canopy) 38%, var(--forest) 78%);
        border-radius: 28px; padding: 38px 42px 34px; color: var(--paper); }
.hero-text { position: relative; z-index: 1; max-width: 63%; }
.hero h1 { color: #F6FAF3 !important; font-weight: 600; font-size: clamp(2rem, 3.1vw, 3.3rem); line-height: 1.04;
           letter-spacing: -0.02em; margin: 0 0 14px; padding: 0; font-variation-settings: 'SOFT' 50; }
.hero .lede { color: #DAE8DD; font-size: 1.08rem; line-height: 1.55; max-width: 48ch; margin: 0 0 20px; }
.detects { display: flex; flex-wrap: wrap; gap: 8px; margin: 0; }
.detects span { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px;
                background: rgba(255,255,255,0.07); border: 1px solid rgba(188,216,195,0.25); color: #F3F6F0; font-size: 0.92rem; }
.detects i { width: 9px; height: 9px; border-radius: 50%; background: var(--c); box-shadow: 0 0 0 3px rgba(255,255,255,0.12); }
.hero .art { position: absolute; right: 8px; bottom: -6px; width: 34%; max-width: 360px; pointer-events: none; }
.art svg { width: 100%; height: auto; }
.art .ln { fill: none; stroke: var(--mint); stroke-width: 1.5; stroke-linecap: round; stroke-dasharray: 1; stroke-dashoffset: 1;
           animation: draw 1.8s cubic-bezier(.45,.05,.2,1) forwards; }
.art .blade { fill: rgba(188,216,195,0.08); opacity: 0; animation: fadein 1.2s ease 1.6s forwards; }
.art .spot { fill: var(--tomato); opacity: 0; animation: fadein .6s ease 2.3s forwards; }
.art .scan { animation: scanloop 5s ease-in-out 2.6s infinite; opacity: 0; }
@keyframes draw { to { stroke-dashoffset: 0; } }
@keyframes fadein { to { opacity: 1; } }
@keyframes scanloop { 0% { transform: translateY(0); opacity: 0; } 12% { opacity: 1; } 70% { opacity: 1; } 85%, 100% { transform: translateY(300px); opacity: 0; } }
@media (max-width: 1200px) { .hero-text { max-width: 100%; } .hero .art { display: none; } }
@media (max-width: 760px) { .hero { padding: 28px 22px; border-radius: 22px; min-height: 0; } }

.st-key-upload_card { background: var(--sheet); border: 1px solid var(--rule); border-radius: 28px; padding: 28px 28px 22px;
                      box-shadow: 0 24px 48px -34px rgba(13,42,28,0.5); height: 100%; box-sizing: border-box; }
.st-key-upload_card h3 { margin: 0; padding: 0; font-size: 1.75rem; font-weight: 600; }
.up-sub { color: var(--muted); margin: 6px 0 2px; line-height: 1.45; }
.st-key-upload_card [data-testid="stFileUploaderDropzone"] { background: var(--paper); border: 1.5px dashed #94BFA0; border-radius: 18px; padding: 26px 18px; }
.st-key-upload_card [data-testid="stFileUploaderDropzone"] button { background: var(--tomato) !important; border: none !important;
                      border-radius: 999px !important; padding: 8px 18px !important; }
.st-key-upload_card [data-testid="stFileUploaderDropzone"] button * { color: #FFFFFF !important; }
.st-key-upload_card [data-testid="stPopover"], .st-key-upload_card [data-testid="stPopover"] > div,
.st-key-upload_card [data-testid="stPopover"] button { width: 100%; }
.st-key-upload_card [data-testid="stPopover"] button { border-radius: 999px; border: 1px solid var(--rule); background: var(--sheet); padding: 10px 16px; }
.st-key-upload_card [data-testid="stPopover"] button:hover { border-color: var(--emerald); color: var(--emerald); }
.up-tips { list-style: none; margin: 4px 0 0 !important; padding: 0 !important; display: grid; gap: 7px; }
.up-tips li { position: relative; margin: 0 !important; padding-left: 28px; font-size: 0.95rem; line-height: 1.4; }
.up-tips li::before { content: '✓'; position: absolute; left: 0; top: 0; width: 19px; height: 19px; border-radius: 50%;
                      background: #E3F0E1; color: var(--emerald); font-size: 11px; font-weight: 700; display: grid; place-items: center; }
.up-note { color: var(--muted); font-size: 0.92rem; margin: 12px 0 0; padding-top: 12px; border-top: 1px solid var(--rule); }

/* section heads */
.sec { display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; flex-wrap: wrap; margin: 40px 0 16px; }
.sec h2 { margin: 0; padding: 0; font-size: clamp(1.7rem, 2.6vw, 2.3rem); font-weight: 600; letter-spacing: -0.01em; }
.sec p { margin: 6px 0 0; color: var(--muted); font-size: 1.05rem; max-width: 70ch; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; }
.legend span { display: inline-flex; align-items: center; gap: 7px; color: var(--muted); font-size: 0.95rem; }
.legend i { width: 10px; height: 10px; border-radius: 50%; background: var(--c); }

/* condition gallery */
.st-key-gallery [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 18px; }
.st-key-gallery [data-testid="stColumn"] { flex: 1 1 0; min-width: 0; }
@media (max-width: 1100px) { .st-key-gallery [data-testid="stColumn"] { flex: 1 1 calc(50% - 18px); min-width: calc(50% - 18px); } }
@media (max-width: 560px) { .st-key-gallery [data-testid="stColumn"] { flex: 1 1 100%; min-width: 100%; } }
[class*="st-key-cond_"] { background: var(--sheet); border: 1px solid var(--rule); border-radius: 22px; padding: 0 0 16px; overflow: hidden;
                          transition: transform .2s ease, box-shadow .2s ease; height: 100%; }
[class*="st-key-cond_"]:hover { transform: translateY(-3px); box-shadow: 0 18px 36px -22px rgba(13,42,28,0.45); }
[class*="st-key-cond_"] .stButton { padding: 0 18px; }
[class*="st-key-test_"], [class*="st-key-test_"] > div, [class*="st-key-test_"] button { width: 100% !important; }
[class*="st-key-cond_"] .stButton button { width: 100%; border-radius: 999px; background: var(--paper); border: 1px solid var(--rule); color: var(--ink); }
[class*="st-key-cond_"] .stButton button:hover { background: var(--forest); color: #fff; border-color: var(--forest); }
[class*="st-key-cond_"] .stButton button:focus-visible { outline: 3px solid var(--leaf); outline-offset: 2px; }
.cc { --c: #2E7D3A; }
.cc-img { position: relative; aspect-ratio: 4 / 3; overflow: hidden; background: #E8EEE5; }
.cc-img img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform .5s ease; }
[class*="st-key-cond_"]:hover .cc-img img { transform: scale(1.04); }
.cc-badge { position: absolute; left: 12px; top: 12px; display: inline-flex; align-items: center; gap: 7px; padding: 6px 11px;
            border-radius: 999px; background: rgba(255,255,255,0.92); color: var(--c); font-weight: 700; font-size: 0.85rem; }
.cc-badge i { width: 8px; height: 8px; border-radius: 50%; background: var(--c); }
.cc-body { padding: 16px 18px 6px; border-top: 4px solid var(--c); }
.cc-body h4 { margin: 0 0 4px; padding: 0; font-size: 1.35rem; font-weight: 600; }
.cc-cause { margin: 0 0 10px; color: var(--muted); font-size: 0.92rem; line-height: 1.4; min-height: 2.6em; }
.cc-look { margin: 0 0 10px; font-size: 0.97rem; line-height: 1.45; min-height: 4.4em; }
.cc-look b { display: block; font-size: 0.85rem; color: var(--muted); font-weight: 400; margin-bottom: 2px; }
.cc-urg { margin: 0; font-size: 0.9rem; font-weight: 700; color: var(--c); }

/* results */
.st-key-results { scroll-margin-top: 12px; }
.live { background: var(--sheet); border: 1px solid var(--rule); border-radius: 22px; padding: 18px 22px; margin-bottom: 16px; }
.live-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; font-weight: 700; }
.live-head .pulse { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: var(--tomato); margin-right: 8px;
                    animation: pulse 1s ease-in-out infinite, settle .3s ease 2.9s forwards; }
.live-head .done { color: var(--muted); font-weight: 400; font-size: 0.92rem; opacity: 0; animation: fadein .4s ease 2.9s forwards; }
.steps { list-style: none; margin: 0 !important; padding: 0 !important; display: grid; grid-template-columns: 1fr 1fr; gap: 8px 18px; }
.steps li { position: relative; padding-left: 30px; margin: 0 !important; color: #9AA89F; font-size: 0.95rem; line-height: 1.35;
            animation: litup .3s ease var(--d) forwards; }
.steps li::before { content: ''; position: absolute; left: 0; top: 1px; width: 16px; height: 16px; border-radius: 50%;
                    border: 2px solid #CFDACB; border-top-color: var(--emerald); animation: spin .7s linear infinite, gone .15s ease var(--d) forwards; }
.steps li::after { content: '✓'; position: absolute; left: 0; top: 0; width: 20px; height: 20px; border-radius: 50%; background: var(--emerald);
                   color: #fff; font-size: 12px; display: grid; place-items: center; opacity: 0; transform: scale(.4);
                   animation: pop .3s ease var(--d) forwards; }
@keyframes litup { to { color: var(--ink); } }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes gone { to { opacity: 0; } }
@keyframes pop { to { opacity: 1; transform: scale(1); } }
@keyframes pulse { 50% { opacity: .35; } }
@keyframes settle { to { background: var(--emerald); animation-play-state: paused; } }
@media (max-width: 560px) { .steps { grid-template-columns: 1fr; } }

.dx { background: var(--sheet); border: 1px solid var(--rule); border-radius: 26px; padding: 26px 28px 22px;
      box-shadow: 0 24px 48px -30px rgba(13,42,28,0.45); opacity: 0; transform: translateY(10px); animation: rise .6s ease 2.4s forwards; }
@keyframes rise { to { opacity: 1; transform: none; } }
.dx-alert { --state: var(--tomato); --state-text: var(--tomato-text); }
.dx-unsure { --state: var(--sun); --state-text: var(--sun-text); }
.dx-ok { --state: var(--leaf); --state-text: var(--leaf-text); }
.dx-top { display: grid; grid-template-columns: 150px minmax(0, 1fr); gap: 24px; align-items: center; }
.gauge { position: relative; width: 150px; height: 150px; border-radius: 50%; background: conic-gradient(var(--state) calc(var(--pa) * 1%), #E4EAE1 0); }
.gauge::after { content: ''; position: absolute; inset: 14px; border-radius: 50%; background: var(--sheet); }
.gauge .val { position: absolute; inset: 0; z-index: 1; display: grid; place-content: center; text-align: center; }
.gauge .val b { font-family: var(--display); font-size: 2.3rem; line-height: 1; font-weight: 600; }
.gauge .val span { font-size: 0.82rem; color: var(--muted); margin-top: 4px; }
.gauge .tick { position: absolute; inset: -7px; z-index: 2; }
.gauge .tick::before { content: ''; position: absolute; left: 50%; top: 0; width: 3px; height: 28px; border-radius: 2px; background: var(--ink); transform: translateX(-50%); }
.dx-group { display: inline-flex; align-items: center; gap: 7px; font-size: 0.88rem; font-weight: 700; color: var(--c); margin: 0 0 4px; }
.dx-group i { width: 8px; height: 8px; border-radius: 50%; background: var(--c); }
.dx-state { margin: 0; font-weight: 700; color: var(--state-text); }
.dx h2 { margin: 2px 0 6px; padding: 0; font-size: clamp(1.9rem, 2.8vw, 2.6rem); line-height: 1.05; font-weight: 600; letter-spacing: -0.01em; }
.dx-kind { margin: 0; color: var(--muted); line-height: 1.45; }
.dx-urgency { display: flex; align-items: center; gap: 10px; margin: 18px 0 0; padding: 12px 16px; border-radius: 14px;
              background: color-mix(in srgb, var(--state) 13%, #fff); color: var(--state-text); font-weight: 700; }
.dx-urgency i { width: 10px; height: 10px; border-radius: 50%; background: var(--state); box-shadow: 0 0 0 5px color-mix(in srgb, var(--state) 25%, transparent); }
.dx-diff { margin-top: 18px; }
.dx-diff p { margin: 0 0 8px; color: var(--muted); font-size: 0.93rem; }
.prob { display: grid; grid-template-columns: minmax(0, 190px) minmax(0, 1fr) 48px; gap: 12px; align-items: center; margin: 7px 0; font-size: 0.97rem; }
.prob .name { display: flex; align-items: center; gap: 8px; }
.prob .name i { flex: none; width: 8px; height: 8px; border-radius: 50%; background: var(--c); }
.prob .track { height: 9px; border-radius: 5px; background: #E4EAE1; overflow: hidden; }
.prob .fill { display: block; height: 100%; border-radius: 5px; background: var(--c); opacity: .55; transform-origin: left; animation: grow 1s cubic-bezier(.2,.8,.2,1) both; }
.prob.top .fill { opacity: 1; }
.prob .pct { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
.prob.top .pct, .prob.top .name { color: var(--ink); font-weight: 700; }
.dx-rule { margin: 14px 0 0; font-size: 0.9rem; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 12px; }
@keyframes grow { from { transform: scaleX(0); } }
@media (max-width: 560px) { .dx-top { grid-template-columns: 1fr; } .prob { grid-template-columns: minmax(0, 130px) minmax(0, 1fr) 42px; } }

/* advice */
.adv { display: grid; grid-template-columns: repeat(var(--cols), minmax(0, 1fr)); gap: 16px; }
.adv section { background: var(--sheet); border: 1px solid var(--rule); border-radius: 20px; padding: 18px 20px; }
.adv h4 { margin: 0 0 8px; padding: 0; font-size: 1.2rem; font-weight: 600; display: flex; align-items: center; gap: 10px; }
.adv h4 i { width: 28px; height: 28px; border-radius: 9px; display: grid; place-items: center; font-style: normal; font-size: 15px;
            background: color-mix(in srgb, var(--c) 14%, #fff); color: var(--c); font-family: var(--body); font-weight: 700; }
.adv ul { margin: 0 0 0 1.1em !important; padding: 0 !important; }
.adv li { margin: 0 0 6px !important; line-height: 1.45; }
.warn { background: var(--sun-soft); border-radius: 12px; padding: 10px 12px; font-size: 0.88rem; margin: 0 0 10px; line-height: 1.4; }
.spray { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
.spray td { border-bottom: 1px solid var(--rule); padding: 8px 4px; vertical-align: top; }
.spray td:last-child { white-space: nowrap; text-align: right; color: var(--muted); padding-left: 10px; }
.expert { background: var(--forest); color: #EAF2EC; border-radius: 20px; padding: 18px 22px; margin-top: 16px; display: flex; gap: 16px; align-items: center; }
.expert b { color: #fff; font-family: var(--display); font-size: 1.15rem; font-weight: 600; white-space: nowrap; }
.sources { font-size: 0.88rem; color: var(--muted); margin-top: 10px; }
.sources a { color: var(--muted); }
.tips { background: var(--sun-soft); border-radius: 18px; padding: 16px 20px; margin-top: 14px; }
.tips ul { margin: 6px 0 0 1.1em !important; padding: 0 !important; }
@media (max-width: 1100px) { .adv { grid-template-columns: 1fr 1fr; } }
@media (max-width: 640px) { .adv { grid-template-columns: 1fr; } .expert { flex-direction: column; align-items: flex-start; } }

/* how it works */
.pipe { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0; margin: 0 0 22px !important; padding: 0 !important;
        list-style: none; counter-reset: s; background: var(--sheet); border: 1px solid var(--rule); border-radius: 22px; overflow: hidden; }
.pipe li { margin: 0 !important; padding: 20px 20px 22px; counter-increment: s; border-right: 1px solid var(--rule); }
.pipe li:last-child { border-right: none; }
.pipe li::before { content: counter(s); display: grid; place-items: center; width: 30px; height: 30px; border-radius: 50%;
                   background: var(--forest); color: #fff; font-size: 0.9rem; margin-bottom: 10px; }
.pipe li b { display: block; font-family: var(--display); font-size: 1.15rem; margin-bottom: 4px; font-weight: 600; }
.pipe li span { color: var(--muted); font-size: 0.95rem; line-height: 1.45; }
.perf { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
.perf div { background: var(--sheet); border: 1px solid var(--rule); border-radius: 20px; padding: 18px 20px; border-top: 4px solid var(--c); }
.perf b { display: block; font-family: var(--display); font-size: 2.2rem; font-weight: 600; line-height: 1.1; }
.perf span { display: block; font-weight: 700; margin: 4px 0; }
.perf small { color: var(--muted); font-size: 0.92rem; line-height: 1.45; display: block; }
@media (max-width: 900px) { .pipe { grid-template-columns: 1fr 1fr; } .pipe li { border-bottom: 1px solid var(--rule); } .perf { grid-template-columns: 1fr; } }
.foot { color: var(--muted); font-size: 0.9rem; border-top: 1px solid var(--rule); margin-top: 34px; padding-top: 16px; line-height: 1.6; }
@media (prefers-reduced-motion: reduce) {
  .art .ln { animation: none; stroke-dashoffset: 0; } .art .blade, .art .spot { animation: none; opacity: 1; } .art .scan { display: none; }
  .steps li, .steps li::before, .steps li::after, .dx, .prob .fill, .live-head .pulse, .live-head .done { animation-duration: 0s !important; animation-delay: 0s !important; }
  .gauge { animation-duration: 0s !important; animation-delay: 0s !important; }
}
</style>
""".replace('__FONTS__', FONTS)
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ model
@st.cache_resource(show_spinner='Loading the model…')
def load_model():
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'final_model.keras'))
    info = json.load(open(os.path.join(MODEL_DIR, 'final_model_info.json')))
    gap = next(l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))
    feature_model = tf.keras.Model(model.input, gap.input)        # photo -> last 7 x 7 feature map
    head = model.layers[model.layers.index(gap):-1]
    return model, info, feature_model, head


@st.cache_data(show_spinner=False)
def load_cards():
    return json.load(open(os.path.join(APP_DIR, 'action_cards.json')))


def b64(image, quality=88):
    buf = io.BytesIO()
    (image if isinstance(image, Image.Image) else Image.fromarray(image)).convert('RGB').save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False)
def thumb_b64(file):
    path = os.path.join(SAMPLE_DIR, 'thumbs', file)
    im = Image.open(path if os.path.exists(path) else os.path.join(SAMPLE_DIR, file))
    return b64(ImageOps.fit(im.convert('RGB'), (480, 360), Image.LANCZOS), 84)


@st.cache_data(show_spinner='Reading the leaf…', max_entries=64)
def analyse(data):
    """Photo bytes -> probabilities, photo and heatmap (base64 JPEG), the raw 7 x 7 Grad-CAM grid and the time taken."""
    model, info, feature_model, head = load_model()
    start = time.perf_counter()
    size = info['img_size']
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
    x = np.clip(np.round(tf.image.resize(np.asarray(image), (size, size)).numpy()), 0, 255).astype(np.float32)
    conv = feature_model(x[None], training=False)
    dense = model.layers[-1]
    with tf.GradientTape() as tape:
        tape.watch(conv)
        h = conv
        for layer in head:
            h = layer(h, training=False)
        logits = tf.matmul(h, dense.kernel) + dense.bias
        score = logits[0, int(tf.argmax(logits[0]))]
    grads = tape.gradient(score, conv)
    cam = tf.nn.relu(tf.reduce_sum(conv * tf.reduce_mean(grads, axis=(1, 2), keepdims=True), axis=-1))[0]
    cam = (cam / (tf.reduce_max(cam) + 1e-8)).numpy()
    probs = tf.nn.softmax(logits[0]).numpy()
    seconds = time.perf_counter() - start
    scale = 900 / max(image.size)
    photo = np.asarray(image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS))
    heat = tf.image.resize(cam[..., None], photo.shape[:2])[..., 0].numpy().clip(0, 1)
    alpha = (0.78 * heat ** 1.15)[..., None]
    overlay = np.clip(photo / 255.0 * (1 - alpha) + colormaps['inferno'](heat)[..., :3] * alpha, 0, 1)
    return probs, b64(photo), b64((overlay * 255).astype(np.uint8)), np.round(cam, 3).tolist(), seconds


def viewer(photo, heat, grid):
    """Scan, show the real 7 x 7 Grad-CAM grid cell by cell, sweep in the heatmap; then drag to compare."""
    components.html("""
<!doctype html><html><head><link rel="stylesheet" href="__FONTS__"><style>
html, body { margin: 0; background: transparent; font-family: 'Atkinson Hyperlegible', system-ui, sans-serif; }
.v { position: relative; height: 616px; border-radius: 26px; overflow: hidden; background: #0D2A1C; user-select: none; touch-action: none; cursor: ew-resize; --x: 100%; }
.v img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; pointer-events: none; }
.v .heat { clip-path: inset(0 0 0 var(--x)); }
.layer { position: absolute; pointer-events: none; }
.scanline { position: absolute; left: 0; right: 0; height: 70px; background: linear-gradient(180deg, rgba(150,255,190,0), rgba(150,255,190,.35) 85%, rgba(220,255,230,.95));
            border-bottom: 2px solid #E6FFEE; opacity: 0; }
.grid { display: grid; grid-template-columns: repeat(7, 1fr); grid-template-rows: repeat(7, 1fr); opacity: 1; transition: opacity .5s ease; }
.grid div { border: 1px solid rgba(255,255,255,.18); opacity: 0; transform: scale(.6); transition: opacity .25s ease, transform .25s ease; }
.line { position: absolute; top: 0; bottom: 0; left: var(--x); width: 2px; background: #F3F6F0; transform: translateX(-1px); box-shadow: 0 0 18px rgba(243,246,240,.8); }
.knob { position: absolute; top: 50%; left: var(--x); width: 50px; height: 50px; border-radius: 50%; background: #F3F6F0; transform: translate(-50%,-50%);
        display: grid; place-items: center; box-shadow: 0 6px 18px rgba(0,0,0,.35); color: #0D2A1C; font-size: 18px; letter-spacing: 2px; outline: none; }
.knob:focus-visible { box-shadow: 0 0 0 4px #5BA44A, 0 6px 18px rgba(0,0,0,.35); }
.tag { position: absolute; top: 16px; padding: 8px 14px; border-radius: 999px; font-size: 14px; color: #F3F6F0; background: rgba(13,42,28,.65); backdrop-filter: blur(8px); }
.tag.l { left: 16px; } .tag.r { right: 16px; }
.cap { position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%); padding: 9px 16px; border-radius: 999px; font-size: 14px; color: #F3F6F0;
       background: rgba(13,42,28,.72); backdrop-filter: blur(8px); white-space: nowrap; display: flex; align-items: center; gap: 9px; }
.cap i { width: 8px; height: 8px; border-radius: 50%; background: #E0452F; animation: p 1s ease-in-out infinite; }
.cap.done i { background: #5BA44A; animation: none; }
@keyframes p { 50% { opacity: .3; } }
.legend { position: absolute; bottom: 16px; right: 16px; display: flex; align-items: center; gap: 8px; font-size: 12px; color: #F3F6F0;
          padding: 7px 11px; border-radius: 999px; background: rgba(13,42,28,.65); }
.legend b { width: 70px; height: 8px; border-radius: 4px; background: linear-gradient(90deg, #000004, #781c6d, #ed6925, #fcffa4); }
@media (max-width: 640px) { .legend { display: none; } }
</style></head><body>
<div class="v" id="v" role="group" aria-label="Compare your photo with the heatmap of where the model looked">
  <img id="base" src="data:image/jpeg;base64,__PHOTO__" alt="Your leaf photo">
  <img class="heat" src="data:image/jpeg;base64,__HEAT__" alt="Heatmap of where the model looked">
  <div class="layer" id="frame"><div class="scanline" id="scan"></div><div class="layer grid" id="grid" style="inset:0"></div></div>
  <div class="line"></div>
  <div class="knob" id="k" tabindex="0" role="slider" aria-label="Divider position" aria-valuemin="0" aria-valuemax="100">‹ ›</div>
  <span class="tag l">Your photo</span><span class="tag r">Where the AI looked</span>
  <span class="cap" id="cap"><i></i><span id="capt">Scanning the leaf</span></span>
  <span class="legend">low <b></b> high</span>
</div>
<script>
const GRID = __GRID__;
const v = document.getElementById('v'), k = document.getElementById('k'), base = document.getElementById('base');
const frame = document.getElementById('frame'), scan = document.getElementById('scan'), gridEl = document.getElementById('grid');
const cap = document.getElementById('cap'), capt = document.getElementById('capt');
const stops = [[0,0,4],[87,16,110],[188,55,84],[249,142,9],[252,255,164]];
const color = t => { const s = Math.min(3.999, t * 4), i = Math.floor(s), f = s - i, a = stops[i], b = stops[i + 1];
  return `rgb(${a.map((c, j) => Math.round(c + (b[j] - c) * f)).join(',')})`; };
let x = 100;
const set = p => { x = Math.max(0, Math.min(100, p)); v.style.setProperty('--x', x + '%'); k.setAttribute('aria-valuenow', Math.round(x)); };
function place() {
  const W = v.clientWidth, H = v.clientHeight, r = (base.naturalWidth || 1) / (base.naturalHeight || 1);
  let w = W, h = W / r; if (h > H) { h = H; w = H * r; }
  Object.assign(frame.style, { left: (W - w) / 2 + 'px', top: (H - h) / 2 + 'px', width: w + 'px', height: h + 'px' });
}
GRID.flat().forEach(val => { const c = document.createElement('div'); c.style.background = color(val);
  c.dataset.o = (0.12 + 0.72 * val).toFixed(2); gridEl.appendChild(c); });
const cells = [...gridEl.children];
const finish = () => { gridEl.style.opacity = 0; set(50); capt.textContent = 'Drag to compare'; cap.classList.add('done'); };
function run() {
  place();
  try { window.frameElement && window.frameElement.scrollIntoView({ behavior: 'smooth', block: 'center' }); } catch (e) {}
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) { finish(); return; }
  const h = frame.clientHeight; let t0;
  scan.style.opacity = 1;
  const sweep = ts => { t0 = t0 || ts; const t = Math.min(1, (ts - t0) / 900); scan.style.top = (t * (h + 70) - 70) + 'px';
    if (t < 1) requestAnimationFrame(sweep); else { scan.style.opacity = 0; grid(); } };
  requestAnimationFrame(sweep);
  function grid() {
    capt.textContent = 'Reading 1,280 features on a 7 × 7 grid';
    cells.forEach((c, i) => setTimeout(() => { c.style.opacity = c.dataset.o; c.style.transform = 'none'; }, i * 22));
    setTimeout(reveal, 49 * 22 + 300);
  }
  function reveal() {
    capt.textContent = 'Marking where it looked';
    gridEl.style.opacity = 0; let t1;
    const go = ts => { t1 = t1 || ts; const t = Math.min(1, (ts - t1) / 900), e = 1 - Math.pow(1 - t, 3); set(100 - 50 * e);
      if (t < 1) requestAnimationFrame(go); else { capt.textContent = 'Drag to compare'; cap.classList.add('done'); } };
    requestAnimationFrame(go);
  }
}
base.complete ? run() : base.addEventListener('load', run);
window.addEventListener('resize', place);
const fromEvent = e => { const r = v.getBoundingClientRect(); return (e.clientX - r.left) / r.width * 100; };
let drag = false;
v.addEventListener('pointerdown', e => { drag = true; v.setPointerCapture(e.pointerId); set(fromEvent(e)); });
v.addEventListener('pointermove', e => { if (drag) set(fromEvent(e)); });
v.addEventListener('pointerup', () => drag = false);
k.addEventListener('keydown', e => { if (e.key === 'ArrowLeft') set(x - 5); if (e.key === 'ArrowRight') set(x + 5); });
</script></body></html>""".replace('__FONTS__', FONTS).replace('__PHOTO__', photo).replace('__HEAT__', heat)
                    .replace('__GRID__', json.dumps(grid)), height=622)


LEAF_SVG = """
<svg viewBox="0 0 380 330" aria-hidden="true">
  <defs><linearGradient id="scanfill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#BCD8C3" stop-opacity="0"/>
  <stop offset="1" stop-color="#BCD8C3" stop-opacity=".22"/></linearGradient></defs>
  <path class="ln" pathLength="1" d="M34 306 C 110 246, 190 166, 330 50"/>
  __LEAFLETS__
  <circle class="spot" cx="226" cy="194" r="5"/><circle class="spot" cx="240" cy="182" r="3.5"/><circle class="spot" cx="218" cy="178" r="2.5"/>
  <g class="scan"><rect x="40" y="18" width="300" height="22" fill="url(#scanfill)"/><rect x="40" y="39" width="300" height="1.5" fill="#E6FFEE" opacity=".8"/></g>
</svg>"""
LEAFLET = ('<g transform="translate({x} {y}) rotate({r}) scale({s})"><path class="blade" d="M0 0 C 18 -24, 66 -28, 98 0 C 66 28, 18 24, 0 0 Z"/>'
           '<path class="ln" pathLength="1" style="animation-delay:{d1}s" d="M0 0 C 18 -24, 66 -28, 98 0 C 66 28, 18 24, 0 0 Z"/>'
           '<path class="ln" pathLength="1" style="animation-delay:{d2}s" d="M4 0 Q 50 -3 94 0"/></g>')
LEAF_SVG = LEAF_SVG.replace('__LEAFLETS__', ''.join(LEAFLET.format(x=x, y=y, r=r, s=s, d1=d, d2=round(d + .35, 2)) for x, y, r, s, d in [
    (112, 244, -98, 1, .35), (118, 239, 12, 1, .45), (186, 176, -96, 1.08, .6), (192, 172, 14, 1.08, .7),
    (258, 110, -92, .92, .85), (262, 106, 18, .92, .95), (326, 54, -40, 1.12, 1.1)]))
LOGO = ('<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="18" r="11" fill="#E0452F"/>'
        '<path d="M16 7c-2-3-6-3-7-1 3 0 5 1 7 3 2-2 4-3 7-3-1-2-5-2-7 1z" fill="#1F7A4A"/></svg>')

# ------------------------------------------------------------------ data
try:
    model, info, _, _ = load_model()
except (FileNotFoundError, OSError, ValueError) as err:
    st.error(f'The model could not be loaded from `{MODEL_DIR}`. Add `final_model.keras` and `final_model_info.json` '
             f'from Drive (TomatoAI/models) to the `models` folder. Details: {err}')
    st.stop()
cards = load_cards()
meta, CLASSES = cards['_meta'], info['classes']
THRESHOLD = float(info.get('uncertain_threshold', 0.5))
ORDER = [c for c in EXAMPLE if c in CLASSES] + [c for c in CLASSES if c not in EXAMPLE]   # grouped: diseases, virus, pest, deficiencies, healthy
GALLERY = [c for c in ORDER if c in EXAMPLE and os.path.exists(os.path.join(SAMPLE_DIR, EXAMPLE[c]))]
if os.path.exists(os.path.join(SAMPLE_DIR, HARD_CASE)):
    GALLERY.append('_hard')

if 'source' not in st.session_state:
    st.session_state.source, st.session_state.example = None, None


def use(source):
    st.session_state.source = source


def pick(name):
    st.session_state.source, st.session_state.example = 'example', name


def group(c):
    return GROUP.get(c, ('Condition', '#536459'))


# ------------------------------------------------------------------ top bar and hero
st.markdown(f"<div class='bar-top'><div class='brand'>{LOGO}Tomato Leaf Check</div>"
            "<div class='who'>Group 23 mini project, B.Tech AI/ML, Universal AI University</div></div>", unsafe_allow_html=True)
with st.container(key='top'):
    hero_col, upload_col = st.columns([1.65, 1], gap='medium')
    with hero_col:
        chips = ''.join(f"<span><i style='--c:{group(c)[1]}'></i>{escape(cards[c]['title'])}</span>" for c in ORDER)
        st.markdown(f"<section class='hero'><div class='hero-text'><h1>Show it a tomato leaf. It tells you what the plant needs.</h1>"
                    f"<p class='lede'>In about a second the AI names the most likely problem, shows exactly where it looked, "
                    f"and tells you what to do. When it is not sure, it says so.</p><div class='detects'>{chips}</div></div>"
                    f"<div class='art'>{LEAF_SVG}</div></section>", unsafe_allow_html=True)
    with upload_col, st.container(key='upload_card'):
        st.markdown("<h3>Check a leaf</h3><p class='up-sub'>Upload a clear photo of one tomato leaf, or take one with your camera.</p>",
                    unsafe_allow_html=True)
        uploaded = st.file_uploader('Upload a leaf photo', type=['jpg', 'jpeg', 'png', 'webp'], key='upload',
                                    on_change=use, args=('upload',), label_visibility='collapsed')
        with st.popover('Take a photo with your camera'):
            snapped = st.camera_input('Leaf photo', key='camera', on_change=use, args=('camera',), label_visibility='collapsed')
        st.markdown("<ul class='up-tips'><li>One leaf, filling most of the frame</li><li>Daylight, without harsh shadows</li>"
                    "<li>A plain background helps</li></ul>"
                    "<p class='up-note'>No photo at hand? Test one of the example leaves further down.</p>", unsafe_allow_html=True)

data = None
if st.session_state.source == 'example' and st.session_state.example in GALLERY:
    name = st.session_state.example
    data = open(os.path.join(SAMPLE_DIR, HARD_CASE if name == '_hard' else EXAMPLE[name]), 'rb').read()
elif st.session_state.source == 'upload' and uploaded is not None:
    data = uploaded.getvalue()
elif st.session_state.source == 'camera' and snapped is not None:
    data = snapped.getvalue()


# ------------------------------------------------------------------ results
def advice_html(name, compact=False):
    card, c = cards[name], group(name)[1]
    blocks = [('!', 'Do this now', '<ul>' + ''.join(f'<li>{escape(t)}</li>' for t in card['act_now']) + '</ul>')]
    if card['sprays'] and not compact:
        rows = ''.join(f"<tr><td>{escape(s['product'])}</td><td>{escape(s['dose'])}</td></tr>" for s in card['sprays'])
        blocks.append(('✚', 'Spray options', f"<div class='warn'>{escape(meta['spray_warning'])}</div><table class='spray'>{rows}</table>"))
    if not compact:
        blocks.append(('↻', 'Prevent it next time', '<ul>' + ''.join(f'<li>{escape(t)}</li>' for t in card['prevent']) + '</ul>'))
    blocks.append(('?', 'Signs to confirm', '<ul>' + ''.join(f'<li>{escape(t)}</li>' for t in card['signs']) + '</ul>'))
    cols = ''.join(f"<section><h4><i>{icon}</i>{title}</h4>{body}</section>" for icon, title, body in blocks)
    return f"<div class='adv' style='--cols:{len(blocks)}; --c:{c}'>{cols}</div>"


if data is not None:
    try:
        probs, photo_b64, heat_b64, grid, seconds = analyse(data)
    except Exception:
        st.error('This file could not be read as a photo. Use a JPG or PNG image of one leaf.')
        st.stop()
    order = np.argsort(-probs)
    top, conf = CLASSES[order[0]], float(probs[order[0]])
    sure = conf >= THRESHOLD
    state = 'unsure' if not sure else ('ok' if top == 'Healthy' else 'alert')
    card = cards[top]

    with st.container(key='results'):
        st.markdown("<div class='sec'><div><h2>What it found</h2><p>Drag the handle on the photo to compare it with the heatmap.</p></div></div>",
                    unsafe_allow_html=True)
        left, right = st.columns([1.3, 1], gap='large')
        with left:
            viewer(photo_b64, heat_b64, grid)
        with right:
            line = 'above' if sure else 'below'
            steps = [('0.15s', f"Photo resized to {info['img_size']} × {info['img_size']}"), ('0.9s', '1,280 features read by MobileNetV2'),
                     ('1.5s', f'{len(CLASSES)} conditions scored'), ('2.0s', f'Confidence {conf:.0%}, {line} the {THRESHOLD:.0%} line'),
                     ('2.3s', 'Heatmap of where it looked')]
            st.markdown("<div class='live'><div class='live-head'><span><span class='pulse'></span>Live analysis</span>"
                        f"<span class='done'>Done in {seconds:.2f} s</span></div><ol class='steps'>"
                        + ''.join(f"<li style='--d:{d}'>{escape(t)}</li>" for d, t in steps) + "</ol></div>", unsafe_allow_html=True)
            g_name, g_col = group(top)
            if sure:
                state_text = 'No problem found' if top == 'Healthy' else 'Most likely'
                title = card['title']
                kind = f"{card['kind']} ({card['cause']})" if card['cause'] else card['signs'][0]
                urgency = f"<div class='dx-urgency'><i></i>{URGENCY[card['urgency']]}</div>"
                group_line = f"<p class='dx-group' style='--c:{g_col}'><i></i>{g_name}</p>"
            else:
                state_text = 'Not sure'
                title = f"{cards[CLASSES[order[0]]]['title']} or {cards[CLASSES[order[1]]]['title'].lower()}?"
                kind, urgency, group_line = cards['_uncertain']['message'], '', ''
            bars = ''.join(
                f"<div class='prob{' top' if j == 0 else ''}' style='--c:{group(CLASSES[i])[1]}'><span class='name'><i></i>{escape(cards[CLASSES[i]]['title'])}</span>"
                f"<span class='track'><span class='fill' style='width:{probs[i] * 100:.1f}%; animation-delay:{1.5 + 0.12 * j:.2f}s'></span></span>"
                f"<span class='pct'>{probs[i]:.0%}</span></div>" for j, i in enumerate(order[:5]))
            p = round(conf * 100, 1)
            ring = f'ring{int(p * 10)}'
            st.markdown(f"""
<div class="dx dx-{state}">
  <style>@keyframes {ring} {{ from {{ --pa: 0; }} to {{ --pa: {p}; }} }}</style>
  <div class="dx-top">
    <div class="gauge" style="--pa:{p}; animation: {ring} 1.2s cubic-bezier(.2,.8,.2,1) 2.5s both;">
      <div class="val"><b>{conf:.0%}</b><span>confident</span></div>
      <div class="tick" style="transform: rotate({THRESHOLD * 360:.1f}deg)"></div>
    </div>
    <div>{group_line}<p class="dx-state">{state_text}</p><h2>{escape(title)}</h2><p class="dx-kind">{escape(kind)}</p></div>
  </div>
  {urgency}
  <div class="dx-diff"><p>How the closest possibilities compare</p>{bars}</div>
  <p class="dx-rule">The dark tick on the ring marks {THRESHOLD:.0%}. Below it, the app says “not sure” instead of naming a problem.</p>
</div>""", unsafe_allow_html=True)
            if not sure:
                tips = cards['_uncertain']
                st.markdown(f"<div class='tips'><b>For a clearer answer</b><ul>{''.join(f'<li>{escape(t)}</li>' for t in tips['retake_tips'])}"
                            "</ul></div>", unsafe_allow_html=True)

        if sure:
            st.markdown(f"<div class='sec'><div><h2>What to do</h2><p>{escape(card['title'])}: {escape(URGENCY[card['urgency']].lower())}.</p></div></div>"
                        + advice_html(top), unsafe_allow_html=True)
            expert_text, source_keys = card['expert'], card['sources']
        else:
            st.markdown("<div class='sec'><div><h2>If it is one of the two closest matches</h2>"
                        "<p>Check these signs on your plant to tell them apart before you act.</p></div></div>", unsafe_allow_html=True)
            a, b = st.columns(2, gap='large')
            for col, i in zip((a, b), order[:2]):
                with col:
                    st.markdown(f"<h3 style='margin:0 0 10px'>If it is {escape(cards[CLASSES[i]]['title'].lower())}</h3>"
                                + advice_html(CLASSES[i], compact=True), unsafe_allow_html=True)
            expert_text = cards['_uncertain']['expert']
            source_keys = list(dict.fromkeys(cards[CLASSES[order[0]]]['sources'] + cards[CLASSES[order[1]]]['sources']))
        links = '; '.join(f"<a href='{meta['sources'][k]['url']}' target='_blank'>{escape(meta['sources'][k]['title'])}</a>"
                          for k in source_keys)
        st.markdown(f"<div class='expert'><b>When to call an expert</b><span>{escape(expert_text)}</span></div>"
                    f"<div class='sources'>Advice from: {links}</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ what it can detect
legend = ''.join(f"<span><i style='--c:{c}'></i>{g}</span>" for g, c in dict(GROUP.values()).items())
st.markdown(f"<div class='sec'><div><h2>What it can detect</h2><p>{len(CLASSES)} conditions of tomato leaves. "
            f"Click any card to test the app on a real photo it never saw during training.</p></div>"
            f"<div class='legend'>{legend}</div></div>", unsafe_allow_html=True)
with st.container(key='gallery'):
    for row in range(0, len(GALLERY), 4):
        for col, (i, name) in zip(st.columns(4, gap='medium'), list(enumerate(GALLERY))[row:row + 4]):
            with col, st.container(key=f'cond_{i}'):
                if name == '_hard':
                    color, badge, title = '#536459', 'Test case', 'A hard field photo'
                    cause = 'Late blight photographed in a real field (PlantDoc, CC BY 4.0).'
                    look, urg = 'Shows how the app answers when a photo is unlike its training photos.', 'Expect “not sure”'
                    thumb = thumb_b64(HARD_CASE)
                else:
                    c = cards[name]
                    badge, color = group(name)
                    title, cause = c['title'], (c['cause'] or 'No disease, pest or deficiency')
                    look, urg = c['signs'][0], URGENCY[c['urgency']]
                    thumb = thumb_b64(EXAMPLE[name])
                st.markdown(f"""<div class="cc" style="--c:{color}">
  <div class="cc-img"><img src="data:image/jpeg;base64,{thumb}" alt="{escape(title)} example"><span class="cc-badge"><i></i>{badge}</span></div>
  <div class="cc-body"><h4>{escape(title)}</h4><p class="cc-cause">{escape(cause)}</p>
  <p class="cc-look"><b>Look for</b>{escape(look)}</p><p class="cc-urg">{escape(urg)}</p></div></div>""", unsafe_allow_html=True)
                st.button('Test this leaf', key=f'test_{i}', on_click=pick, args=(name,))

# ------------------------------------------------------------------ how it works
lab_acc, lab_f1 = info.get('test_accuracy'), info.get('test_macro_f1')
field_f1, pd_acc = info.get('field_cv_macro_f1'), info.get('plantdoc_accuracy')
perf = []
if lab_acc is not None:
    perf.append(('#2E7D3A', f'{lab_acc:.0%}', 'correct on lab photos', f'316 test photos of single leaves it never saw. Macro-F1 {lab_f1:.2f}.'))
if field_f1 is not None:
    perf.append(('#B26A00', f'{field_f1:.2f}', 'macro-F1 on field photos', 'Leaves from real Rajasthan fields, tested on photo blocks it never trained on.'))
if pd_acc is not None:
    perf.append(('#B8392A', f'{pd_acc:.0%}', 'correct on internet photos', 'Other countries and cameras (PlantDoc). Not reliable there, so it often says “not sure”.'))
st.markdown(f"""
<div class="sec"><div><h2>How it works, and how well</h2><p>The model is a {escape(MODEL_TEXT.get(info['config'], info['config']))}.</p></div></div>
<ol class="pipe">
  <li><b>Photo</b><span>Your photo is resized to {info['img_size']} × {info['img_size']} pixels.</span></li>
  <li><b>Features</b><span>MobileNetV2, pre-trained on 1.2 million everyday photos, describes it with 1,280 numbers on a 7 × 7 grid.</span></li>
  <li><b>Scores</b><span>A logistic-regression classifier turns them into a probability for each of the {len(CLASSES)} conditions.</span></li>
  <li><b>Confidence check</b><span>Below {THRESHOLD:.0%} it answers “not sure”. Answered photos are right about 80% of the time.</span></li>
  <li><b>Heatmap</b><span>Grad-CAM marks the parts of the photo that pushed the answer most.</span></li>
</ol>
<div class="perf">{''.join(f"<div style='--c:{c}'><b>{v}</b><span>{l}</span><small>{escape(s)}</small></div>" for c, v, l, s in perf)}</div>
<div class="foot">Built by Group 23, B.Tech AI/ML, Universal AI University, under the guidance of Mr. Rajesh Bhise.<br>
Data: Tomato-Village (Gehlot, Saxena and Gandhi, Multimedia Systems, 2023) and PlantDoc (Singh et al., 2020).
Advice: TNAU Agritech Portal and the AESA-based IPM package for tomato (NIPHM, Government of India). Always confirm sprays with your nearest Krishi Vigyan Kendra.</div>
""", unsafe_allow_html=True)
