let keystrokeData = [];

document.addEventListener('keydown', (event) => {
    keystrokeData.push({
        key: event.key,
        event: 'keydown',
        timestamp: Date.now()
    });
});

document.addEventListener('keyup', (event) => {
    keystrokeData.push({
        key: event.key,
        event: 'keyup',
        timestamp: Date.now()
    });
});

function sendData() {
    const features = processKeystrokeData(keystrokeData);

    fetch('/identify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features }),
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('result').innerHTML = `Identified User: ${data.user}`;
        })
        .catch(error => console.error('Error:', error));
}

function processKeystrokeData(data) {
    // Example feature extraction: Calculate press duration and flight times
    let features = [];
    for (let i = 1; i < data.length; i++) {
        if (data[i].event === 'keyup' && data[i - 1].event === 'keydown') {
            features.push(data[i].timestamp - data[i - 1].timestamp); // Press duration
        }
    }
    return features;
}
