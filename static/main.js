document.addEventListener('DOMContentLoaded', () => {

    // --- 1. UI ELEMENTS ---
    const chatWidget = document.getElementById('chat-widget');
    const chatToggleBtn = document.getElementById('chat-toggle-btn');
    const closeChatBtn = document.getElementById('close-chat-btn');
    const messagesContainer = document.getElementById('messages-container');
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');

    let conversationHistory = [];
    let hasWelcomed = false;

    // --- 2. OPEN/CLOSE LOGIC ---
    function toggleChat() {
        chatWidget.classList.toggle('d-none');
        if (!hasWelcomed && !chatWidget.classList.contains('d-none')) {
            appendMessage('ai', 'أهلاً بيك في سيتى فوم!  أنا المساعد الذكي، إزاي أقدر أساعدك تختار المرتبة أو المخدة المناسبة ليك النهاردة؟<br> Welcome to CityFoam! I’m your smart assistant, here to help you find the perfect mattress or pillow. How can I assist you today?');
            hasWelcomed = true;
        }
    }

    chatToggleBtn.addEventListener('click', toggleChat);
    closeChatBtn.addEventListener('click', () => chatWidget.classList.add('d-none'));

    // --- 3. CHAT API LOGIC ---
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const userMessage = messageInput.value.trim();
        if (!userMessage) return;

        appendMessage('user', userMessage);
        messageInput.value = '';

        const typingDiv = document.createElement('div');
        typingDiv.className = 'message ai typing-indicator';
        typingDiv.innerHTML = '<span></span><span></span><span></span>';
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query: userMessage,
                    history: conversationHistory
                })
            });

            const data = await response.json();
            messagesContainer.removeChild(typingDiv);

            if (response.ok && data.status === 'success') {
                const botMessage = data.response;
                const interactionId = data.interaction_id;
                appendMessage('ai', botMessage, interactionId);

                conversationHistory.push({ role: 'user', content: userMessage });
                conversationHistory.push({ role: 'assistant', content: botMessage });

                if (conversationHistory.length > 6) {
                    conversationHistory = conversationHistory.slice(conversationHistory.length - 6);
                }
            } else {
                appendMessage('ai', `Error: ${data.detail || 'Failed to get response'}`);
            }

        } catch (error) {
            console.error('Fetch error:', error);
            if (messagesContainer.contains(typingDiv)) messagesContainer.removeChild(typingDiv);
            appendMessage('ai', 'Network error. Make sure the backend is running.');
        }
    });

    // --- 4. MESSAGE DRAWING LOGIC (with guaranteed star insertion) ---
    function appendMessage(sender, rawText, interactionId = null) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);
        messagesContainer.appendChild(messageDiv);

        if (sender === 'ai') {
            let htmlText = rawText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            htmlText = htmlText.replace(/\n/g, '<br>');

            let i = 0;
            const typingSpeed = 15;

            function typeWriter() {
                if (i < htmlText.length) {
                    if (htmlText.charAt(i) === '<') {
                        let tagEnd = htmlText.indexOf('>', i);
                        if (tagEnd !== -1) {
                            i = tagEnd + 1;
                        }
                    }
                    messageDiv.innerHTML = htmlText.substring(0, i);
                    i++;
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                    setTimeout(typeWriter, typingSpeed);
                } else {
                    // Bookmark: typing completely finished, now append stars if needed
                    messageDiv.innerHTML = htmlText; // ensure final state
                    if (interactionId) {
                        addRatingStars(messageDiv, interactionId);
                    }
                }
            }
            typeWriter();

        } else {
            messageDiv.textContent = rawText;
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }

    // --- 5. STAR RATING FUNCTIONALITY (with persistence) ---
    function addRatingStars(container, interactionId) {
        // Avoid duplicate stars
        if (container.querySelector('.rating-stars')) return;

        const starsDiv = document.createElement('div');
        starsDiv.className = 'rating-stars';
        starsDiv.setAttribute('data-interaction-id', interactionId);

        for (let i = 1; i <= 5; i++) {
            const star = document.createElement('span');
            star.innerHTML = '★';
            star.dataset.value = i;
            star.addEventListener('click', () => rateResponse(interactionId, i, starsDiv));
            star.addEventListener('mouseenter', () => {
                // Allow hover only if not rated yet
                if (!starsDiv.classList.contains('rated')) {
                    highlightStars(starsDiv, i);
                }
            });
            star.addEventListener('mouseleave', () => {
                if (!starsDiv.classList.contains('rated')) {
                    resetStars(starsDiv);
                }
            });
            starsDiv.appendChild(star);
        }
        container.appendChild(starsDiv);
        console.log('Stars added for interaction', interactionId);
    }

    function highlightStars(container, value) {
        const stars = container.querySelectorAll('span');
        stars.forEach((star, index) => {
            star.classList.toggle('active', index < value);
        });
    }

    function resetStars(container) {
        const stars = container.querySelectorAll('span');
        stars.forEach(star => star.classList.remove('active'));
    }

    async function rateResponse(interactionId, rating, starsContainer) {
        const satisfaction = rating >= 4 ? 1 : 0;
        try {
            const res = await fetch(`/api/rate?interaction_id=${interactionId}&rating=${satisfaction}`, {
                method: 'POST'
            });
            const data = await res.json();
            if (res.ok) {
                // Permanently lock stars
                starsContainer.classList.add('rated');
                starsContainer.style.pointerEvents = 'none';
                highlightStars(starsContainer, rating);
                // Show small confirmation text
                const confirm = document.createElement('small');
                confirm.className = 'rating-confirm';
                confirm.textContent = ' ✓ تم التقييم';
                starsContainer.appendChild(confirm);
                console.log('Rating saved:', data.message);
            } else {
                alert('فشل تسجيل التقييم');
            }
        } catch (err) {
            alert('خطأ في الاتصال أثناء التقييم');
        }
    }

});