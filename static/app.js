(async () => {
    const MODEL_URL = "model.onnx";
    // 1) init webcam
    const video = document.getElementById("cam");
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
    });
    video.srcObject = stream;

    // 2) load model
    const session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ["wasm"],
    });

    // helper: softmax
    function softmax(arr) {
        const max = Math.max(...arr);
        const exps = arr.map((x) => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map((x) => x / sum);
    }

    // 3) capture + run
    document.getElementById("capture").onclick = async () => {
        const canvas = document.getElementById("buf");
        const ctx = canvas.getContext("2d");

        const dataURL = canvas.toDataURL("image/png");
        document.getElementById("captured").src = dataURL;

        // draw current frame
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const { data, width, height } = ctx.getImageData(
            0,
            0,
            canvas.width,
            canvas.height
        );

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
        const logits = outputs.logits.data; // Float32Array
        const probs = softmax(Array.from(logits));
        const pred = probs.indexOf(Math.max(...probs));

        console.log("logits:", Array.from(logits));
        console.log("probs:", probs);
        console.log("predicted_class_index:", pred);
    };
})();
