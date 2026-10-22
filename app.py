"""Tomato Leaf Check: Streamlit app for the Group 23 mini project
"AI-Based Early Detection and Management System for Crop Diseases and Pest Infestations".

Run locally:   streamlit run app.py
Needs, next to this file:
  models/final_model.keras, models/final_model_info.json   (from Drive: TomatoAI/models)
  action_cards.json                                          (advice for each class)
  samples/*.jpg                                              (optional sample leaves)
"""
import io
import os
import json
from html import escape

import numpy as np
import streamlit as st
import tensorflow as tf
from matplotlib import colormaps
from PIL import Image, ImageOps

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(APP_DIR, 'models'))
SAMPLE_DIR = os.path.join(APP_DIR, 'samples')
SAMPLES = [('Early blight', 'early_blight.jpg'), ('Late blight', 'late_blight.jpg'), ('Spotted wilt', 'spotted_wilt.jpg'),
           ('Leaf miner', 'leaf_miner.jpg'), ('Magnesium deficiency', 'magnesium_deficiency.jpg'),
           ('Nitrogen deficiency', 'nitrogen_deficiency.jpg'), ('Healthy', 'healthy.jpg'),
           ('Field photo (hard)', 'field_photo_late_blight.jpg')]
SAMPLES = [(label, f) for label, f in SAMPLES if os.path.exists(os.path.join(SAMPLE_DIR, f))]
MODEL_TEXT = {
    'mobilenetv2_probe': 'Frozen MobileNetV2 with a logistic-regression classifier, trained on lab photos of single leaves.',
    'mobilenetv2_probe_field': 'Frozen MobileNetV2 with a logistic-regression classifier, trained on lab photos and on leaves cut from Tomato-Village field photos.',
    'mobilenetv2_all': 'MobileNetV2 fine-tuned on lab photos of single leaves.',
}
URGENCY = {'high': 'Act within days.', 'medium': 'Act this week.', 'low': 'Not urgent: correct it at the next feeding.',
           'none': 'Nothing to do for this leaf.'}

st.set_page_config(page_title='Tomato Leaf Check', page_icon='🍅', layout='wide', initial_sidebar_state='collapsed')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bitter:wght@500;600;700&family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap');
:root {
  --paper: #F3F6F1; --sheet: #FCFDFB; --ink: #1E3A2B; --muted: #56685C; --rule: #D5DED3;
  --tomato: #C0392B; --tomato-soft: #F7E3DF; --amber: #8A5A12; --amber-fill: #D69E2E; --amber-soft: #FBF0D9;
  --vine: #366E32; --vine-fill: #4E8B3A; --vine-soft: #E3EFDF;
}
[data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"], .stTabs, .stAlert { font-family: 'Atkinson Hyperlegible', system-ui, sans-serif; }
[data-testid="stAppViewContainer"], [data-testid="stApp"] { background: var(--paper); color: var(--ink); }
[data-testid="stMainBlockContainer"], .block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stToolbar"], #MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
h1, h2, h3, h4 { font-family: 'Bitter', Georgia, serif !important; color: var(--ink); letter-spacing: -0.01em; }

.hero { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 24px; align-items: end;
        border-bottom: 2px solid var(--ink); padding-bottom: 18px; margin-bottom: 8px; }
.hero h1 { font-size: clamp(2.4rem, 5vw, 3.6rem); line-height: 1.02; margin: 0; padding: 0; font-weight: 700; }
.hero p { font-size: 1.12rem; line-height: 1.5; color: var(--muted); max-width: 62ch; margin: 10px 0 0; }
.hero .credit { text-align: right; font-size: 0.9rem; color: var(--muted); line-height: 1.4; }
@media (max-width: 720px) { .hero { grid-template-columns: 1fr; } .hero .credit { text-align: left; } }

.classes { color: var(--muted); font-size: 0.98rem; margin: 12px 0 6px; max-width: 80ch; }

.empty { border: 1.5px dashed #B9C6B5; border-radius: 6px; padding: 28px 24px; color: var(--muted); background: var(--sheet); }
.empty strong { color: var(--ink); font-family: 'Bitter', Georgia, serif; font-size: 1.2rem; display: block; margin-bottom: 6px; }

.slip { background: var(--sheet); border: 1px solid var(--rule); border-left: 8px solid var(--state);
        border-radius: 4px; padding: 22px 26px 20px; }
.slip .state { font-size: 1rem; color: var(--state-text); font-weight: 700; margin: 0; }
.slip h2 { font-size: clamp(1.9rem, 3.6vw, 2.6rem); line-height: 1.08; margin: 4px 0 6px; padding: 0; font-weight: 700; }
.slip .cause { color: var(--muted); margin: 0 0 18px; line-height: 1.45; }
.slip .urgency { font-weight: 700; color: var(--state-text); margin: 14px 0 0; }
.slip .alts { color: var(--muted); margin: 6px 0 0; font-size: 0.98rem; }
.state-alert { --state: var(--tomato); --state-text: var(--tomato); }
.state-unsure { --state: var(--amber-fill); --state-text: var(--amber); }
.state-ok { --state: var(--vine-fill); --state-text: var(--vine); }

.meter { position: relative; margin-top: 4px; }
.meter .track { position: relative; height: 14px; border-radius: 7px; background: #E4EAE1; overflow: visible; }
.meter .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 7px; background: var(--state);
               animation: grow 0.9s cubic-bezier(.2,.8,.2,1) both; transform-origin: left center; }
.meter .mark { position: absolute; top: -7px; bottom: -7px; width: 0; border-left: 2px dashed var(--ink); }
.meter .legend { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; margin-top: 10px; font-size: 0.92rem; color: var(--muted); }
.meter .legend span:first-child { white-space: nowrap; font-size: 1rem; }
.meter .legend span:last-child { text-align: right; }
.meter .legend b { color: var(--ink); font-size: 1.05rem; }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@media (prefers-reduced-motion: reduce) { .meter .fill { animation: none; } }

.tips { background: var(--amber-soft); border-radius: 4px; padding: 14px 18px; margin-top: 14px; color: var(--ink); }
.tips ul { margin: 6px 0 0 1.1em; padding: 0; }
.warn { background: var(--amber-soft); color: var(--ink); border-radius: 4px; padding: 12px 16px; font-size: 0.95rem; margin-bottom: 12px; }
.expert { border-top: 1px solid var(--rule); margin-top: 18px; padding-top: 12px; }
.sources { font-size: 0.88rem; color: var(--muted); margin-top: 8px; }
.sources a { color: var(--muted); }
.figure-note { color: var(--muted); font-size: 0.92rem; margin-top: 6px; }
.howto ol { padding-left: 1.2em; } .howto li { margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)


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


@st.cache_data(show_spinner='Looking at the leaf…', max_entries=64)
def analyse(data):
    """Photo bytes -> (class probabilities, heatmap 0..1 at 224 x 224, display photo)."""
    model, info, feature_model, head = load_model()
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
    scale = 760 / max(image.size)       # show small lab photos large and big phone photos small
    display = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    return tf.nn.softmax(logits[0]).numpy(), cam, np.asarray(display)


def heat_overlay(photo, cam):
    h, w = photo.shape[:2]
    heat = tf.image.resize(cam[..., None], (h, w))[..., 0].numpy().clip(0, 1)
    alpha = (0.75 * heat ** 1.2)[..., None]
    color = colormaps['inferno'](heat)[..., :3]
    return np.clip(photo / 255.0 * (1 - alpha) + color * alpha, 0, 1)


def bullets(items):
    return '\n'.join(f'- {i}' for i in items)


def card_body(card, meta, compact=False):
    tabs = ['What to do now', 'Spray options', 'Prevention', 'Signs to check'] if card['sprays'] else \
           ['What to do now', 'Prevention', 'Signs to check']
    panes = dict(zip(tabs, st.tabs(tabs)))
    with panes['What to do now']:
        st.markdown(bullets(card['act_now']))
    if 'Spray options' in panes:
        with panes['Spray options']:
            st.markdown(f"<div class='warn'>{escape(meta['spray_warning'])}</div>", unsafe_allow_html=True)
            st.markdown('| Product | Dose |\n|---|---|\n' + '\n'.join(f"| {s['product']} | {s['dose']} |" for s in card['sprays']))
    with panes['Prevention']:
        st.markdown(bullets(card['prevent']))
    with panes['Signs to check']:
        st.markdown(bullets(card['signs']))
    links = '; '.join(f"<a href='{meta['sources'][k]['url']}' target='_blank'>{escape(meta['sources'][k]['title'])}</a>"
                      for k in card['sources'])
    st.markdown(f"<div class='expert'><b>When to call an expert:</b> {escape(card['expert'])}</div>"
                f"<div class='sources'>Advice from: {links}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- page
try:
    model, info, _, _ = load_model()
except (FileNotFoundError, OSError, ValueError) as err:
    st.error(f'The model could not be loaded from `{MODEL_DIR}`. Add `final_model.keras` and `final_model_info.json` '
             f'from Drive (TomatoAI/models) to the `models` folder. Details: {err}')
    st.stop()
cards = load_cards()
meta, CLASSES = cards['_meta'], info['classes']
THRESHOLD = float(info.get('uncertain_threshold', 0.5))

st.markdown(f"""
<div class="hero">
  <div>
    <h1>Tomato leaf check</h1>
    <p>Photograph one tomato leaf. The app names the most likely disease, pest or deficiency, shows where it looked,
    and tells you what to do next. When it is not confident, it says so.</p>
  </div>
  <div class="credit">Group 23 mini project<br>B.Tech AI/ML, Universal AI University</div>
</div>
<p class="classes">It recognises {len(CLASSES)} cases: {escape(', '.join(cards[c]['title'].lower() if c != 'Healthy' else 'healthy leaves' for c in CLASSES))}.</p>
""", unsafe_allow_html=True)

if 'source' not in st.session_state:
    st.session_state.source = None


def use(source):
    st.session_state.source = source


tab_upload, tab_camera, tab_sample = st.tabs(['Upload a photo', 'Use the camera', 'Try a sample leaf'])
with tab_upload:
    uploaded = st.file_uploader('Leaf photo', type=['jpg', 'jpeg', 'png', 'webp'], key='upload',
                                on_change=use, args=('upload',), label_visibility='collapsed')
with tab_camera:
    snapped = st.camera_input('Leaf photo', key='camera', on_change=use, args=('camera',), label_visibility='collapsed')
with tab_sample:
    picked = st.pills('Sample leaf', [s[0] for s in SAMPLES], key='sample', on_change=use, args=('sample',),
                      label_visibility='collapsed') if SAMPLES else None
    st.caption('Test photos the model never saw during training, from the Tomato-Village dataset. '
               'The field photo comes from PlantDoc (CC BY 4.0) and shows how the app behaves on a hard case.')

sources = {'upload': uploaded, 'camera': snapped,
           'sample': open(os.path.join(SAMPLE_DIR, dict(SAMPLES)[picked]), 'rb') if picked else None}
chosen = sources.get(st.session_state.source) or uploaded or snapped or sources['sample']

if chosen is None:
    st.markdown("<div class='empty'><strong>No photo yet</strong>Upload a photo of one leaf, take one with your camera, "
                "or open “Try a sample leaf” to see the app on a test photo.</div>", unsafe_allow_html=True)
else:
    try:
        probs, cam, photo = analyse(chosen.getvalue() if hasattr(chosen, 'getvalue') else chosen.read())
    except Exception:
        st.error('This file could not be read as a photo. Use a JPG or PNG image of one leaf.')
        st.stop()
    order = np.argsort(-probs)
    top, conf = CLASSES[order[0]], float(probs[order[0]])
    sure = conf >= THRESHOLD
    state = 'unsure' if not sure else ('ok' if top == 'Healthy' else 'alert')

    left, right = st.columns([1.05, 1], gap='large')
    with left:
        view = st.segmented_control('View', ['Where it looked', 'Photo'], default='Where it looked',
                                    key='view', label_visibility='collapsed') or 'Where it looked'
        st.image(heat_overlay(photo, cam) if view == 'Where it looked' else photo)
        if view == 'Where it looked':
            st.markdown("<div class='figure-note'>Brighter areas pushed the answer most (Grad-CAM).</div>",
                        unsafe_allow_html=True)
    with right:
        card = cards[top]
        if sure:
            state_text = 'No problem found' if top == 'Healthy' else 'Most likely'
            title, cause = card['title'], f"{card['kind']}." + (f" {card['cause']}." if card['cause'] else '')
            urgency = URGENCY[card['urgency']]
        else:
            state_text = 'Not sure'
            title = f"Could be {cards[CLASSES[order[0]]]['title'].lower()} or {cards[CLASSES[order[1]]]['title'].lower()}"
            cause = cards['_uncertain']['message']
            urgency = ''
        others = [f"{escape(cards[CLASSES[i]]['title'])} {probs[i]:.0%}" for i in order[(1 if sure else 0):(3 if sure else 2)]]
        alt_label = 'Next closest' if sure else 'Closest matches'
        st.markdown(f"""
<div class="slip state-{state}">
  <p class="state">{state_text}</p>
  <h2>{escape(title)}</h2>
  <p class="cause">{escape(cause)}</p>
  <div class="meter">
    <div class="track"><div class="fill" style="width:{conf * 100:.1f}%"></div>
      <div class="mark" style="left:{THRESHOLD * 100:.1f}%"></div></div>
    <div class="legend"><span>Confidence <b>{conf:.0%}</b></span>
      <span>The app names a problem only above {THRESHOLD:.0%} (dashed line)</span></div>
  </div>
  {f'<p class="urgency">{urgency}</p>' if urgency else ''}
  <p class="alts">{alt_label}: {', '.join(others)}</p>
</div>""", unsafe_allow_html=True)
        if not sure:
            tips = cards['_uncertain']
            st.markdown(f"<div class='tips'><b>For a clearer answer</b><ul>{''.join(f'<li>{escape(t)}</li>' for t in tips['retake_tips'])}"
                        f"</ul><p style='margin:8px 0 0'>{escape(tips['expert'])}</p></div>", unsafe_allow_html=True)

    st.markdown('### What to do' if sure else '### If it is one of the closest matches')
    if sure:
        card_body(cards[top], meta)
    else:
        for c, tab in zip(order[:2], st.tabs([f"If it is {cards[CLASSES[i]]['title']}" for i in order[:2]])):
            with tab:
                card_body(cards[CLASSES[c]], meta)

with st.expander('How it works, and how well'):
    lab_acc, lab_f1 = info.get('test_accuracy'), info.get('test_macro_f1')
    st.markdown(f"""
<div class="howto">
<ol>
<li>Your photo is resized to {info['img_size']} × {info['img_size']} pixels.</li>
<li>MobileNetV2, a network trained on 1.2 million everyday photos (ImageNet), turns it into 1,280 numbers that describe its shapes, colours and textures.</li>
<li>A logistic-regression classifier, trained on our photos, turns those numbers into a probability for each of the {len(CLASSES)} classes.</li>
<li>If the top probability is below {THRESHOLD:.0%}, the app says “Not sure” instead of guessing. We chose this line so that answered photos are right about 80% of the time.</li>
<li>Grad-CAM marks the parts of the photo that pushed the answer most.</li>
</ol>
<p><b>Model:</b> {escape(MODEL_TEXT.get(info['config'], info['config']))}</p>
{f"<p><b>On 316 lab test photos it never saw:</b> {lab_acc:.0%} correct (macro-F1 {lab_f1:.2f}).</p>" if lab_acc else ''}
<p><b>Limits:</b> {escape(meta['app_warning'])}</p>
<p class="sources">Data: Tomato-Village (Gehlot, Saxena and Gandhi, Multimedia Systems, 2023) and PlantDoc (Singh et al., 2020).
Advice: TNAU Agritech Portal and the AESA-based IPM package for tomato (NIPHM, Government of India).
Built by Group 23, B.Tech AI/ML, Universal AI University, under the guidance of Mr. Rajesh Bhise.</p>
</div>""", unsafe_allow_html=True)
