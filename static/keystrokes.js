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

    // Send data to the server when the form is submitted
    const form = document.getElementById("keystrokeForm");
    form.addEventListener("submit", (e) => {
        e.preventDefault(); // Prevent the form from reloading the page
        fetch("/keystrokes", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(keystrokes)
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            alert("Keystroke data submitted!");
            keystrokes = []; // Clear the keystroke array
            textArea.value = ""; // Clear the text area
        })
        .catch(error => console.error("Error:", error));
    });
});
