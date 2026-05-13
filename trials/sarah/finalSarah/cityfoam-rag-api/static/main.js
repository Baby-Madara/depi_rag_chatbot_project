document.addEventListener('DOMContentLoaded', () => {
    
    const messagesContainer = document.getElementById('messages-container');
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');


    let conversationHistory = [];

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault(); // This stops the webpage from refreshing!
        
        const userMessage = messageInput.value.trim();
        if (!userMessage) return;

        // 1. Display User Message
        appendMessage('user', userMessage);
        messageInput.value = '';

        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot typing-indicator';
        typingDiv.innerHTML = '<div class="content">...</div>';
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    query: userMessage,
                    history: conversationHistory
                })
            });

            const data = await response.json();
            
            // Remove the "typing..." indicator
            messagesContainer.removeChild(typingDiv);
            
            if (response.ok && data.status === 'success') {
                const botMessage = data.response;
                appendMessage('bot', botMessage);
                
                
                conversationHistory.push({ role: 'user', content: userMessage });
                conversationHistory.push({ role: 'assistant', content: botMessage });
                
                
                if (conversationHistory.length > 6) {
                    conversationHistory = conversationHistory.slice(conversationHistory.length - 6);
                }
            } else {
                appendMessage('system', `Error: ${data.detail || 'Failed to get response'}`);
            }

        } catch (error) {
            console.error('Fetch error:', error);
            messagesContainer.removeChild(typingDiv);
            appendMessage('system', 'Network error. Make sure the backend is running.');
        } 
    });

    function appendMessage(sender, text) {
        const messageDiv = document.createElement('div');
        // Your friend's CSS uses 'ai' instead of 'bot', so we adjust it here:
        const cssClass = sender === 'bot' ? 'ai' : sender; 
        messageDiv.classList.add('message', cssClass);
        
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('content');
        contentDiv.textContent = text;
        
        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);
        
        
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
});