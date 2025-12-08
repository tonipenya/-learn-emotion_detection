import { classifyEmotion } from "./emotionClassifier.js";
import { detectFaces } from "./faceDetector.js";

const EMOTION_EMOJIS = [ // Follows order in FER+ csv
    "😐", // neutral
    "😊", // happiness
    "😲", // surprise
    "😭", // sadness
    "😡", // anger
    "🤢", // disgust
    "😱", // fear
    "😒", // contempt
];

async function loop(fn, fpsLimit = 30) {
    const frameDuration = 1000 / fpsLimit;

    while (true) {
        const t0 = performance.now();

        await fn();

        const elapsed = performance.now() - t0;
        const wait = frameDuration - elapsed;
        if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    }
}

function draw(emoji, ctx, box) {
    ctx.save();
    const size = Math.max(box.width, box.height);
    ctx.font = `${size}px system-ui, Apple Color Emoji, Segoe UI Emoji, Noto Color Emoji`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(emoji, box.x0 + box.width / 2, box.y0 + box.height / 2);
    ctx.restore();
}

async function processFrame() {
    const frameTrace = {
        boxes: [],
        boxesImageData: [],
        emotions: [],
    };
    const video = document.getElementById("cam");
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    // Detect faces
    const boxes = await detectFaces(
        ctx.getImageData(0, 0, video.videoWidth, video.videoHeight)
    );
    frameTrace.boxes = boxes;

    // Classify each face
    const emotions = [];
    for (const box of boxes) {
        const imageData = ctx.getImageData(box.x0, box.y0, box.width, box.height);
        const emotion = await classifyEmotion(imageData);
        emotions.push(emotion);
        frameTrace.boxesImageData.push(imageData);
    }
    frameTrace.emotions = emotions;

    // // Draw boxes around faces
    for (let i = 0; i < boxes.length; i++) {
        const box = boxes[i];
        const emotion = await emotions[i];
        draw(EMOTION_EMOJIS[emotion], ctx, box);
    }

    // Update output
    const dataURL = canvas.toDataURL("image/png");
    const img = document.getElementById("captured");
    img.src = dataURL;
    await new Promise((r) => (img.onload = r)); // wait until rendered
    return frameTrace;
}

(async () => {
    const video = document.getElementById("cam");
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
    });
    video.srcObject = stream;

    document.getElementById("capture").onclick = async () => loop(processFrame, 30);

    function debugImageData(imageData) {
        // Create a canvas dynamically
        const canvas = document.createElement("canvas");
        canvas.width = imageData.width;
        canvas.height = imageData.height;
        canvas.style.border = "1px solid #aaa";
        document.body.appendChild(canvas);

        const ctx = canvas.getContext("2d");
        ctx.putImageData(imageData, 0, 0);
    }

    document.getElementById("sample").onclick = async () => {
        const trace = await processFrame();
        debugImageData(trace.boxesImageData[0]);
        console.log(trace);
    };
})();
