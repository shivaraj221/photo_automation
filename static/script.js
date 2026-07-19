document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('file');
    const fileWrapper = document.querySelector('.file-upload-wrapper');
    const submitBtn = document.getElementById('submit-btn');
    const loadingCard = document.getElementById('loading-card');
    const resultsCard = document.getElementById('results-card');
    const uploadText = document.querySelector('.upload-text');

    // Drag and Drop
    fileWrapper.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileWrapper.classList.add('dragover');
    });

    fileWrapper.addEventListener('dragleave', () => {
        fileWrapper.classList.remove('dragover');
    });

    fileWrapper.addEventListener('drop', (e) => {
        e.preventDefault();
        fileWrapper.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            updateFileName();
        }
    });

    fileInput.addEventListener('change', updateFileName);

    function updateFileName() {
        if (fileInput.files.length > 0) {
            uploadText.textContent = fileInput.files[0].name;
            uploadText.style.color = '#818cf8';
        } else {
            uploadText.textContent = 'Drag & Drop or Click to Upload Portrait';
            uploadText.style.color = '#e2e8f0';
        }
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!fileInput.files.length) {
            alert('Please select a photo first.');
            return;
        }

        // UI State
        submitBtn.disabled = true;
        submitBtn.textContent = 'PROCESSING...';
        loadingCard.classList.remove('hidden');
        resultsCard.classList.add('hidden');
        
        // Prepare Data
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('copies', document.getElementById('copies').value);
        formData.append('bg_color_hex', document.getElementById('bg_color_hex').value);
        formData.append('force_ai', document.getElementById('force_ai').checked);

        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                // Update images
                document.getElementById('master-img').src = result.passport_url;
                document.getElementById('sheet-img').src = result.sheet_url;
                
                // Update download links
                document.getElementById('download-master').href = result.passport_url;
                document.getElementById('download-sheet').href = result.sheet_url;

                // Setup local print button
                document.getElementById('print-btn').onclick = () => {
                    const printWindow = window.open('', '_blank');
                    printWindow.document.write(`
                        <html>
                        <head>
                            <title>Print Photo</title>
                            <style>
                                @page { size: 4in 6in; margin: 0; }
                                body { margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; background: white; }
                                img { width: 100%; height: auto; }
                            </style>
                        </head>
                        <body>
                            <img src="${result.sheet_url}" onload="window.print();window.close()">
                        </body>
                        </html>
                    `);
                    printWindow.document.close();
                };

                // Show Results
                loadingCard.classList.add('hidden');
                resultsCard.classList.remove('hidden');
                resultsCard.scrollIntoView({ behavior: 'smooth' });
            } else {
                throw new Error(result.error || 'Unknown error occurred.');
            }
        } catch (error) {
            alert('Pipeline Error: ' + error.message);
            loadingCard.classList.add('hidden');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'INITIALIZE AI PIPELINE';
        }
    });
});
