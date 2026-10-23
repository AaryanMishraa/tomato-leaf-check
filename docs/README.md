# Tomato Leaf Check: offline phone app and "Beat the AI" game

Group 23, B.Tech AI/ML, Universal AI University. Part of the mini project *AI-Based Early Detection and Management System for Crop Diseases and Pest Infestations* (tomato).

The app runs our final model (`mobilenetv2_probe_field`: frozen MobileNetV2 + logistic regression) **on the phone itself**. It works in airplane mode, in English, Hindi and Marathi, and has three tabs:

- **Check a leaf:** take or choose a photo. It shows the diagnosis, a confidence ruler with the 85% "Not sure" line, a Grad-CAM heatmap and the action card (what to do now, sprays, prevention, signs, when to call an expert). It can read the advice aloud.
- **Beat the AI:** 8 leaves, you against the AI. 5 are held-out lab test photos and 3 are internet field photos (PlantDoc). At the end it explains the lab-to-field drop using our real results.
- **How it works:** the pipeline, our results, and a speed test that measures the time per photo on the phone in your hand.

## Put it online (free, about 10 minutes)

1. Unzip `tomato-leaf-check-app.zip`. You get a folder named `tomato-leaf-check` with `index.html` inside.
2. Go to <https://github.com> and sign in, or create a free account.
3. Click **+** (top right), then **New repository**. Name it `tomato-leaf-check`, choose **Public**, and click **Create repository**.
4. On the next page, click the link **uploading an existing file**.
5. Open the unzipped `tomato-leaf-check` folder on your computer, press **Ctrl + A** to select everything inside it, and drag it all onto the GitHub page. Wait until all the files are listed (the model file is 9 MB). Click **Commit changes**.
6. Open **Settings**, then **Pages** in the left menu. Under *Build and deployment*, set **Source** to *Deploy from a branch*, **Branch** to *main* and the folder to */ (root)*. Click **Save**.
7. Wait 1 to 2 minutes and refresh. The page shows *Your site is live at* `https://YOUR-USERNAME.github.io/tomato-leaf-check/`. That is your app's link.

Check that the folders `css`, `data`, `fonts`, `icons`, `js`, `model` and `vendor` appear in the repository. If you see only loose files, the folders were not dragged: drag the folders themselves onto **Add file → Upload files**.

## Install it on a phone

1. Open the link in **Chrome** on Android (or **Safari** on iPhone).
2. Wait until the line under the title says **Works without internet**. The first visit downloads about 14 MB.
3. Android: Chrome menu **⋮ → Add to Home screen → Install** (or the **Install the app** button on the *How it works* tab). iPhone: **Share → Add to Home Screen**.
4. Test it: turn on **airplane mode**, open the app from the home screen and tap a sample leaf.

## A 60-second demo for the viva

1. Turn on airplane mode in front of the examiner. Open the app from the home screen.
2. Tap a sample leaf: the diagnosis, confidence and advice card appear almost at once. Tap **Where the AI looked** to show the Grad-CAM heatmap.
3. Switch to **हिं** or **मरा** and tap **Read aloud**.
4. Tap the **Field photo** sample: the app says **Not sure** instead of guessing.
5. Hand the phone over: "Can you beat our AI?" The results screen explains the lab-to-field drop in our own numbers.
6. On **How it works**, tap **Test the speed** and read out the time per photo on this phone.

Points worth saying: the phone gives the same answers as our Keras model; nothing is sent to a server; the model is 9.0 MB; for our linear head, Grad-CAM equals the class weights times the feature maps, so the heatmap costs almost nothing on the phone.

## How it was checked

- **Same answers as Keras.** For all 77 photos bundled in the app, the phone gives the same top class and the same "Not sure" decision as `final_model.keras`; the largest confidence difference is 0.00005.
- **Same preprocessing.** The app reimplements TensorFlow's bilinear resize (half-pixel centres, float32 arithmetic) and numpy's rounding, so the 224 × 224 input is pixel-identical to the Python app's.
- **Held-out game photos.** The lab photos are a random 8 per class (7 for nitrogen) from our 316-photo clean test split (rebuilt with notebook 01's procedure, seed 42; it reproduces test accuracy 0.699 and macro-F1 0.554). The final model never trained on them. The field photos are a random 5 per class from PlantDoc's 262 tomato photos (early blight, late blight, healthy), skipping photos with stock-photo watermarks, logos or drawings. No photo was picked or dropped because of the AI's answer.
- **Offline and big photos.** Tested with the network switched off, from a sub-folder address like GitHub Pages, and with a 12-megapixel rotated camera photo.

## Numbers for the paper (experiment E5, phone deployment)

| Item | Value |
| --- | --- |
| Download size of the model | 9.0 MB (graph 0.14 MB + weights 8.86 MB, float32) |
| Parameters | 2,266,951 (MobileNetV2 backbone 2,257,984 + head 8,967); batch-norm layers are folded into the convolutions on conversion |
| Whole app, first download | about 14 MB, then fully offline |
| Runtime | TensorFlow.js 4.22, WebAssembly backend (falls back to WebGL or plain JavaScript) |
| Time per photo | Measure on your phones with **Test the speed** and report the phone model next to it |

## What's inside

| Path | What it is |
| --- | --- |
| `index.html`, `css/app.css` | The page and its look |
| `js/app.js` | Screens: check a leaf, the game, how it works |
| `js/model.js` | On-phone inference: resize, MobileNetV2, logistic regression, Grad-CAM heatmap |
| `js/i18n.js` | Interface text in English, Hindi and Marathi |
| `data/cards.json` | The action cards (advice) in three languages, with sources |
| `data/model.json` | Class names, the 85% threshold, our results, and the logistic-regression weights |
| `data/game.json`, `data/photos.json` | The game and sample photos, and which class each one is |
| `model/` | MobileNetV2 converted to TensorFlow.js |
| `vendor/` | TensorFlow.js and its WebAssembly files |
| `sw.js`, `manifest.webmanifest`, `icons/` | What makes it installable and offline |

To change the advice, edit `data/cards.json` on GitHub (open the file, click the pencil, **Commit changes**). Phones pick up the change the next time they open the app with internet.

## Before you show it

- The **Hindi and Marathi text is a draft translation**. Ask a teammate who reads each language to check it, especially the spray doses.
- The spray names and doses come from the TNAU Agritech Portal and the NIPHM AESA-based IPM package for tomato. Have the cards checked by an agronomy faculty member or your KVK, as planned.
- Photos: Tomato-Village (Gehlot, Saxena and Gandhi, *Multimedia Systems* 29, 2023; its repository has no licence file, so this is academic use, confirm with your guide) and PlantDoc (Singh et al., CODS-COMAD 2020; CC BY 4.0). Software: TensorFlow.js (Apache 2.0). Fonts: Bitter, Atkinson Hyperlegible, Noto Sans and Noto Serif Devanagari (SIL Open Font Licence).
