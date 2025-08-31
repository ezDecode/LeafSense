// LeafSense Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const resultsSection = document.getElementById('resultsSection');
    const resultImage = document.getElementById('resultImage');
    const resultsContent = document.getElementById('resultsContent');
    
    // Image preview functionality
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            if (file.size > 10 * 1024 * 1024) { // 10MB limit
                alert('File size must be less than 10MB');
                imageInput.value = '';
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.innerHTML = `
                    <img src="${e.target.result}" alt="Preview" class="img-fluid">
                    <p class="text-muted mt-2">${file.name}</p>
                `;
            };
            reader.readAsDataURL(file);
        }
    });
    
    // Form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const file = imageInput.files[0];
        if (!file) {
            alert('Please select an image first');
            return;
        }
        
        // Show loading modal
        const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
        loadingModal.show();
        
        // Disable submit button
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
        
        // Create FormData
        const formData = new FormData();
        formData.append('image', file);
        
        // Send request to backend
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Hide loading modal
            loadingModal.hide();
            
            // Display results
            displayResults(data);
            
            // Show results section
            resultsSection.style.display = 'block';
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            console.error('Error:', error);
            loadingModal.hide();
            
            // Show error message
            alert('An error occurred while processing the image. Please try again.');
        })
        .finally(() => {
            // Re-enable submit button
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="fas fa-search me-2"></i>Detect Disease';
        });
    });
    
    // Function to display results
    function displayResults(data) {
        // Set result image
        resultImage.src = `/static/images/${data.uploaded_image}`;
        
        // Create results content
        let resultsHTML = '';
        
        // Model prediction
        if (data.model_prediction) {
            resultsHTML += `
                <div class="mb-3">
                    <h6 class="text-primary"><i class="fas fa-brain me-2"></i>AI Model Prediction:</h6>
                    <div class="alert alert-info">
                        <strong>Disease:</strong> ${formatDiseaseName(data.model_prediction.disease)}<br>
                        <strong>Confidence:</strong> ${(data.model_prediction.confidence * 100).toFixed(1)}%
                    </div>
                </div>
            `;
        }
        
        // API prediction
        if (data.api_prediction) {
            resultsHTML += `
                <div class="mb-3">
                    <h6 class="text-success"><i class="fas fa-globe me-2"></i>External API Prediction:</h6>
                    <div class="alert alert-success">
                        <strong>Disease:</strong> ${data.api_prediction.disease}<br>
                        <strong>Confidence:</strong> ${(data.api_prediction.confidence * 100).toFixed(1)}%
                        ${data.api_prediction.metadata ? `<br><strong>Source:</strong> ${data.api_prediction.metadata}` : ''}
                    </div>
                </div>
            `;
        }
        
        // Gemini enhanced response
        if (data.gemini_response) {
            resultsHTML += `
                <div class="mb-3">
                    <h6 class="text-warning"><i class="fas fa-magic me-2"></i>AI-Enhanced Analysis:</h6>
                    <div class="alert alert-warning">
                        ${data.gemini_response.replace(/\n/g, '<br>')}
                    </div>
                </div>
            `;
        }
        
        // Timestamp
        if (data.timestamp) {
            resultsHTML += `
                <div class="text-muted small">
                    <i class="fas fa-clock me-1"></i>Analysis completed: ${new Date(data.timestamp).toLocaleString()}
                </div>
            `;
        }
        
        resultsContent.innerHTML = resultsHTML;
    }
    
    // Function to format disease names for better readability
    function formatDiseaseName(diseaseName) {
        if (!diseaseName) return 'Unknown';
        
        // Replace underscores with spaces and capitalize
        return diseaseName
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase())
            .replace(/\b(And|Or|The|A|An)\b/g, l => l.toLowerCase())
            .replace(/\b\w/g, l => l.toUpperCase());
    }
    
    // Function to reset the form
    window.resetForm = function() {
        // Reset form
        uploadForm.reset();
        imagePreview.innerHTML = `
            <i class="fas fa-image fa-3x text-muted"></i>
            <p class="text-muted mt-2">No image selected</p>
        `;
        
        // Hide results section
        resultsSection.style.display = 'none';
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };
    
    // Drag and drop functionality
    imagePreview.addEventListener('dragover', function(e) {
        e.preventDefault();
        imagePreview.style.borderColor = '#28a745';
        imagePreview.style.backgroundColor = '#e8f5e8';
    });
    
    imagePreview.addEventListener('dragleave', function(e) {
        e.preventDefault();
        imagePreview.style.borderColor = '#dee2e6';
        imagePreview.style.backgroundColor = '#f8f9fa';
    });
    
    imagePreview.addEventListener('drop', function(e) {
        e.preventDefault();
        imagePreview.style.borderColor = '#dee2e6';
        imagePreview.style.backgroundColor = '#f8f9fa';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            imageInput.files = files;
            imageInput.dispatchEvent(new Event('change'));
        }
    });
    
    // Click to upload functionality
    imagePreview.addEventListener('click', function() {
        imageInput.click();
    });
    
    // Add visual feedback for file input
    imageInput.addEventListener('focus', function() {
        imagePreview.style.borderColor = '#28a745';
        imagePreview.style.backgroundColor = '#e8f5e8';
    });
    
    imageInput.addEventListener('blur', function() {
        imagePreview.style.borderColor = '#dee2e6';
        imagePreview.style.backgroundColor = '#f8f9fa';
    });
    
    // Add loading state to image preview
    function setImagePreviewLoading() {
        imagePreview.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-success" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="text-muted mt-2">Processing image...</p>
            </div>
        `;
    }
    
    // Add error state to image preview
    function setImagePreviewError(message) {
        imagePreview.innerHTML = `
            <div class="text-center text-danger">
                <i class="fas fa-exclamation-triangle fa-3x"></i>
                <p class="mt-2">${message}</p>
            </div>
        `;
    }
    
    // Keyboard navigation support
    imageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            uploadForm.dispatchEvent(new Event('submit'));
        }
    });
    
    // Accessibility improvements
    imagePreview.setAttribute('tabindex', '0');
    imagePreview.setAttribute('role', 'button');
    imagePreview.setAttribute('aria-label', 'Click or drag and drop to upload image');
    
    // Add success message after successful upload
    function showSuccessMessage(message) {
        const successAlert = document.createElement('div');
        successAlert.className = 'alert alert-success alert-dismissible fade show';
        successAlert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        uploadForm.insertBefore(successAlert, uploadForm.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            successAlert.remove();
        }, 5000);
    }
    
    // Add error message display
    function showErrorMessage(message) {
        const errorAlert = document.createElement('div');
        errorAlert.className = 'alert alert-danger alert-dismissible fade show';
        errorAlert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        uploadForm.insertBefore(errorAlert, uploadForm.firstChild);
        
        // Auto-dismiss after 8 seconds
        setTimeout(() => {
            errorAlert.remove();
        }, 8000);
    }
    
    // Export functions for global access
    window.LeafSense = {
        showSuccessMessage,
        showErrorMessage,
        setImagePreviewLoading,
        setImagePreviewError
    };
});
