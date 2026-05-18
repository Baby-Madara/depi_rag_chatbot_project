document.addEventListener('DOMContentLoaded', () => {
    
    // --- 1. UI ELEMENTS ---
    const chatWidget = document.getElementById('chat-widget');
    const chatToggleBtn = document.getElementById('chat-toggle-btn');
    const closeChatBtn = document.getElementById('close-chat-btn');
    
    const messagesContainer = document.getElementById('messages-container');
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');

    let conversationHistory = [];
    let hasWelcomed = false; // Remembers if we already said hello!

    // --- 2. OPEN/CLOSE LOGIC ---
    function toggleChat() {
        // This toggles the Bootstrap 'd-none' class (display: none)
        chatWidget.classList.toggle('d-none');
        
        // If it's the first time opening, say hello!
        if (!hasWelcomed && !chatWidget.classList.contains('d-none')) {
            appendMessage('ai', 'أهلاً بيك في سيتى فوم!  أنا المساعد الذكي، إزاي أقدر أساعدك تختار المرتبة أو المخدة المناسبة ليك النهاردة؟<br> Welcome to CityFoam! I’m your smart assistant, here to help you find the perfect mattress or pillow. How can I assist you today?');
            hasWelcomed = true;
        }
    }

    // Connect the buttons to the open/close function
    chatToggleBtn.addEventListener('click', toggleChat);
    closeChatBtn.addEventListener('click', () => chatWidget.classList.add('d-none'));


    // --- 3. CHAT API LOGIC ---
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault(); // Stop page refresh
        
        const userMessage = messageInput.value.trim();
        if (!userMessage) return;

        // Display User Message
        appendMessage('user', userMessage);
        messageInput.value = '';

        // Show cool bouncing dots typing indicator
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message ai typing-indicator';
        typingDiv.innerHTML = '<span></span><span></span><span></span>';
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        try {
            // Send query to your Python FastAPI backend
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query: userMessage,
                    history: conversationHistory
                })
            });

            const data = await response.json();
            
            // Remove typing indicator when backend replies
            messagesContainer.removeChild(typingDiv);
            
            if (response.ok && data.status === 'success') {
                const botMessage = data.response;
                appendMessage('ai', botMessage);
                
                // Save to history context
                conversationHistory.push({ role: 'user', content: userMessage });
                conversationHistory.push({ role: 'assistant', content: botMessage });
                
                // Keep history to last 6 messages to avoid token bloat
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


    // --- 4. MESSAGE DRAWING LOGIC ---
    // --- 4. MESSAGE DRAWING LOGIC ---
    function appendMessage(sender, rawText) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);
        messagesContainer.appendChild(messageDiv);
        
        if (sender === 'ai') {
            // 1. Pre-process Markdown into HTML before typing!
            // Convert **text** to <strong>text</strong>
            let htmlText = rawText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            // Convert hidden newlines (\n) to actual HTML line breaks (<br>)
            htmlText = htmlText.replace(/\n/g, '<br>');

            let i = 0;
            const typingSpeed = 15;

            function typeWriter() {
                if (i < htmlText.length) {
                    
                    // 2. The Lookahead Trick: If we hit an HTML tag, skip over it!
                    if (htmlText.charAt(i) === '<') {
                        let tagEnd = htmlText.indexOf('>', i);
                        if (tagEnd !== -1) {
                            i = tagEnd + 1; // Jump past the entire <tag>
                        }
                    }
                    
                    // 3. Draw everything up to the current point
                    messageDiv.innerHTML = htmlText.substring(0, i);
                    
                    i++;
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                    setTimeout(typeWriter, typingSpeed);
                }
            }
            typeWriter();
            
        } else {
            messageDiv.textContent = rawText;
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }
});