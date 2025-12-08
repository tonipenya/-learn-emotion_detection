import { createCanvas, ImageData, loadImage } from "canvas";
import fs from "fs";
import path from "path";

// Load ONNX Runtime for Node and expose it as the global `ort` used by the
// browser-style `emotionClassifier.js` module.
const ortModule = await import("onnxruntime-node");
globalThis.ort = ortModule.default || ortModule;

// Adapt create() so browser-only executionProviders (e.g. "wasm") are removed
// when running under Node with onnxruntime-node. This preserves the existing
// browser-style calls in `emotionClassifier.js` while allowing Node to select
// a suitable native backend.
if (
    globalThis.ort &&
    globalThis.ort.InferenceSession &&
    typeof globalThis.ort.InferenceSession.create === "function"
) {
    const origCreate = globalThis.ort.InferenceSession.create.bind(
        globalThis.ort.InferenceSession
    );
    globalThis.ort.InferenceSession.create = async (modelPath, options = {}) => {
        if (options && options.executionProviders) {
            // Remove executionProviders so onnxruntime-node picks a native backend
            const newOptions = { ...options };
            delete newOptions.executionProviders;
            return origCreate(modelPath, newOptions);
        }
        return origCreate(modelPath, options);
    };
}

// Minimal `document` and `ImageData` so emotionClassifier's DOM calls succeed.
globalThis.ImageData = ImageData;
globalThis.document = {
    createElement: () => createCanvas(1, 1),
    body: { appendChild: () => {} },
};

// Ensure relative model path inside `static` resolves correctly
process.chdir(path.join(process.cwd(), "static"));

// Temporarily silence console INFO logs while importing and running the
// browser-style classifier which uses console.log for debug/info messages.
const __orig_console_log = console.log;
const __orig_console_info = console.info;
const __orig_console_debug = console.debug;
console.log = () => {};
console.info = () => {};
console.debug = () => {};

// Import the browser-style classifier (uses top-level await in the module)
const { classifyEmotion } = await import("./static/emotionClassifier.js");

// Walk validation images and classify
const imagesDir = path.join(process.cwd(), "../data/ferplus_raw/FER2013Test");
const outRows = ["filename,prediction"];

const files = fs.readdirSync(imagesDir);
for (const f of files) {
    const imgPath = path.join(imagesDir, f);
    const img = await loadImage(imgPath);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, img.width, img.height);

    const pred = await classifyEmotion(imageData);
    outRows.push(`${f},${pred}`);
}

fs.writeFileSync(
    path.join(process.cwd(), "../data/node_predictions.csv"),
    outRows.join("\n")
);

// Restore console functions and print final status
console.log = __orig_console_log;
console.info = __orig_console_info;
console.debug = __orig_console_debug;

console.log("Done. Wrote predictions to data/node_predictions.csv");
