let keystrokes = [];

document.addEventListener("DOMContentLoaded", () => {
    const textArea = document.getElementById("inputArea");

    // Capture keydown and keyup events
    textArea.addEventListener("keydown", (event) => {
        const timestamp = Date.now();
        keystrokes.push({
            key: event.key,
            type: "keydown",
            timestamp: timestamp
        });
    });

    textArea.addEventListener("keyup", (event) => {
        const timestamp = Date.now();
        keystrokes.push({
            key: event.key,
            type: "keyup",
            timestamp: timestamp
        });
    });

});
