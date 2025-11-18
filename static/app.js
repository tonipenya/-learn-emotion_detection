import { classifyEmotion } from "./emotionClassifier.js";
import { detectFaces } from "./faceDetector.js";

const BOX_COLORS = [
    "#FF00FF", // angry
    "#00FFFF", // disgust
    "#FF0000", // fear
    "#FFFF00", // happy
    "#FF8800", // neutral
    "#00FF00", // sad
    "#0000FF", // surprise
];

const EMOTION_EMOJIS = [
    "😡", // anger
    "😒", // contempt
    "🤢", // disgust
    "😱", // fear
    "😊", // happiness
    "😊", // neutral
    "😊", // sadness
    "😲", // surprise
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
    // Classify each face
    const emotions = [];
    for (const box of boxes) {
        const emotion = await classifyEmotion(
            ctx.getImageData(box.x0, box.y0, box.width, box.height)
        );
        emotions.push(emotion);
    }

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
}

(async () => {
    const video = document.getElementById("cam");
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
    });
    video.srcObject = stream;

    document.getElementById("capture").onclick = async () => loop(processFrame, 30);
})();
