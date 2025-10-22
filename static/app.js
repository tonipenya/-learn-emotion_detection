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

(async () => {
    const video = document.getElementById("cam");
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
    });
    video.srcObject = stream;

    document.getElementById("capture").onclick = async () => {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Detect faces
        const boxes = await detectFaces(
            ctx.getImageData(0, 0, canvas.width, canvas.height)
        );

        const emotions = [];
        for (const box of boxes) {
            const emotion = await classifyEmotion(
                ctx.getImageData(box.x0, box.y0, box.width, box.height)
            );
            emotions.push(emotion);
        }

        // // Draw boxes around faces
        ctx.lineWidth = 2;

        for (let i = 0; i < boxes.length; i++) {
            const box = boxes[i];
            const emotion = await emotions[i];
            ctx.strokeStyle = BOX_COLORS[emotion];
            ctx.strokeRect(box.x0, box.y0, box.width, box.height);
        }

        const dataURL = canvas.toDataURL("image/png");
        document.getElementById("captured").src = dataURL;
    };
})();
