class Box {
    constructor(x0, y0, x1, y1) {
        this.x0 = x0;
        this.y0 = y0;
        this.x1 = x1;
        this.y1 = y1;
    }

    get width() {
        return Math.abs(this.x1 - this.x0);
    }

    get height() {
        return Math.abs(this.y1 - this.y0);
    }

    get area() {
        return this.width * this.height;
    }
}

function toBoxes(flat) {
    // from [x0, y0, x1, y1, ...] to [Box, Box, ...]
    if (flat.length % 4 !== 0) throw new Error("Odd number of coordinates");
    const boxes = [];
    for (let i = 0; i < flat.length; i += 4) {
        boxes.push(new Box(flat[i], flat[i + 1], flat[i + 2], flat[i + 3]));
    }
    return boxes;
}

function intersectionOverUnion(a, b) {
    const x0 = Math.max(a.x0, b.x0);
    const y0 = Math.max(a.y0, b.y0);
    const x1 = Math.min(a.x1, b.x1);
    const y1 = Math.min(a.y1, b.y1);

    const intersection = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
    const union = a.area + b.area - intersection;
    return intersection === 0 ? 0 : intersection / union;
}

function nms(boxes, scores, threshold, maxBoxes) {
    const indices = scores
        .map((s, i) => [s, i])
        .sort((a, b) => b[0] - a[0])
        .map((x) => x[1]);

    const keep = []; // Boxes
    // iterate over scores
    for (let k = 0; k < indices.length && keep.length < maxBoxes; k++) {
        const currentBoxIndex = indices[k];
        const currentBox = boxes[currentBoxIndex];

        // Iterate over kept boxes
        let shouldKeep = true;
        for (let keptBox of keep) {
            if (intersectionOverUnion(currentBox, keptBox) > threshold) {
                shouldKeep = false;
                break;
            }
        }

        if (shouldKeep) {
            console.log("Keeping box with score:", scores[currentBoxIndex]);
            keep.push(currentBox);
        }
    }
    return keep;
}

function scaled(imageData, width, height) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    canvas.width = width;
    canvas.height = height;

    // Draw original ImageData scaled
    const tmp = document.createElement("canvas");
    tmp.width = imageData.width;
    tmp.height = imageData.height;
    tmp.getContext("2d").putImageData(imageData, 0, 0);

    ctx.drawImage(tmp, 0, 0, width, height);
    return ctx.getImageData(0, 0, width, height);
}

class FaceDetector {
    constructor(session) {
        this.session = session;
    }

    async detectFaces(imageData) {
        // Resize to 320x240
        const inputWidth = 320;
        const inputHeight = 240;
        const data = scaled(imageData, inputWidth, inputHeight).data;

        // Channel-Height-Width representation
        const size = inputWidth * inputHeight;
        let redOffset = 0;
        let greenOffset = size;
        let blueOffset = 2 * size; // 2* to skip alpha
        let chw = new Float32Array(3 * size);
        for (let i = 0; i < size; i++) {
            const j = i * 4;
            chw[redOffset + i] = data[j];
            chw[greenOffset + i] = data[j + 1];
            chw[blueOffset + i] = data[j + 2];
        }

        // normalization according to UltraFace specs
        const mean = 127;
        const scale = 1.0 / 128;
        chw = chw.map((v) => (v - mean) * scale);

        // Build tensor
        const inputName = this.session.inputNames[0];
        const inputTensor = new ort.Tensor("float32", chw, [
            1,
            3,
            inputHeight,
            inputWidth,
        ]);

        // Run inference
        let out = null;
        try {
            out = await this.session.run({ [inputName]: inputTensor });
        } catch (error) {
            console.error("Error during face detection inference:", error);
            return null;
        }

        // Map to regular arrays (those from model are Float32Arrays)
        let boxes = Array.from(out.boxes.data);
        boxes = toBoxes(out.boxes.data);
        let scores = Array.from(out.scores.data);

        // Discard low-confidence boxes
        // Scores are interleaved: [no_face, face, no_face, face, ...]
        let faceProbabilities = scores.filter((_, i) => i % 2 === 1); // keep only face scores
        const indicesToKeep = faceProbabilities
            .map((score, index) => (score > 0.7 ? index : -1))
            .filter((index) => index !== -1);
        boxes = indicesToKeep.map((i) => boxes[i]);
        faceProbabilities = indicesToKeep.map((i) => faceProbabilities[i]);

        boxes = nms(boxes, faceProbabilities, 0.3, 3);

        // Map boxes back to original image size
        boxes = boxes.map(
            (box) =>
                new Box(
                    box.x0 * imageData.width,
                    box.y0 * imageData.height,
                    box.x1 * imageData.width,
                    box.y1 * imageData.height
                )
        );
        return boxes;
    }
}

const MODEL_URL = "ultraface-RFB-320.onnx";
const session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
});
const faceDetector = new FaceDetector(session);
export const detectFaces = faceDetector.detectFaces.bind(faceDetector);
