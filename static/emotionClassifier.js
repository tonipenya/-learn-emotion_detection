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

    async classifyEmotion(imageData) {
        const { data, width, height } = imageData;

        // pack to Float32Array [1,3,H,W] with values in [0,255]
        const size = width * height;
        const tensorData = new Float32Array(3 * size);
        let rOff = 0,
            gOff = size,
            bOff = 2 * size;
        for (let i = 0; i < size; i++) {
            const j = i * 4; // RGBA
            tensorData[rOff + i] = data[j];
            tensorData[gOff + i] = data[j + 1];
            tensorData[bOff + i] = data[j + 2];
        }

        const inputTensor = new ort.Tensor("float32", tensorData, [
            1,
            3,
            height,
            width,
        ]);

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

const MODEL_URL = "assets/basemodel.onnx";
const session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
});
const emotionClassifier = new EmotionClassifier(session);
export const classifyEmotion =
    emotionClassifier.classifyEmotion.bind(emotionClassifier);
