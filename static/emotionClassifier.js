function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map((x) => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((x) => x / sum);
}

class EmotionClassifier {
    constructor(session) {
        this.session = session;
    }

    preprocess(imageData) {
        const imgSize = 48;

        // 1. Resize using canvas
        const srcCanvas = document.createElement("canvas");
        srcCanvas.width = imageData.width;
        srcCanvas.height = imageData.height;
        const sctx = srcCanvas.getContext("2d");
        sctx.putImageData(imageData, 0, 0);

        const dstCanvas = document.createElement("canvas");
        dstCanvas.width = imgSize;
        dstCanvas.height = imgSize;
        const dctx = dstCanvas.getContext("2d");
        dctx.drawImage(srcCanvas, 0, 0, imgSize, imgSize);

        const resized = dctx.getImageData(0, 0, imgSize, imgSize);
        const { data } = resized; // RGBA
        const size = imgSize * imgSize;

        // 2. Grayscale + ToTensor + Normalize
        // output: Float32Array [1, 1, imgSize, imgSize]
        const tensorData = new Float32Array(size); // 1 channel

        for (let i = 0; i < size; i++) {
            const j = i * 4;
            const r = data[j];
            const g = data[j + 1];
            const b = data[j + 2];

            // same grayscale formula as torchvision
            const gray = 0.299 * r + 0.587 * g + 0.114 * b; // [0,255]

            // ToTensor: /255 -> [0,1]
            let x = gray / 255.0;

            // Normalize(mean=0.5, std=0.5): (x - 0.5) / 0.5
            x = (x - 0.5) / 0.5; // -> [-1,1]

            tensorData[i] = x;
        }

        // Pack as [1, 1, H, W] (NCHW)
        return new ort.Tensor("float32", tensorData, [1, 1, imgSize, imgSize]);
    }

    async classifyEmotion(imageData) {
        const inputTensor = this.preprocess(imageData);

        const outputs = await session.run({ input: inputTensor });
        const logits = outputs.logits.data;
        const probs = softmax(Array.from(logits));
        const pred = probs.indexOf(Math.max(...probs));

        console.log("logits:", Array.from(logits));
        console.log("probs:", probs);
        console.log("predicted_class_index:", pred);

        return pred;
    }
}

const MODEL_URL = "./assets/model.onnx";
const session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
});
const emotionClassifier = new EmotionClassifier(session);
export const classifyEmotion =
    emotionClassifier.classifyEmotion.bind(emotionClassifier);
