document.getElementById('upload-form').addEventListener('submit', async (event) => {
    event.preventDefault();

    const filesInput = document.getElementById('files');
    const questionInput = document.getElementById('question');
    const responseDiv = document.getElementById('response');

    const formData = new FormData();
    for (const file of filesInput.files) {
        formData.append('files', file);
    }

    try {
        // First, upload the files
        const uploadResponse = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) {
            throw new Error('File upload failed');
        }

        const uploadResult = await uploadResponse.json();

        // Then, send the question
        const queryResponse = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: questionInput.value
            })
        });

        if (!queryResponse.ok) {
            throw new Error('Query failed');
        }

        const queryResult = await queryResponse.json();
        responseDiv.innerText = queryResult.answer;

    } catch (error) {
        responseDiv.innerText = `Error: ${error.message}`;
    }
});
