document.addEventListener('DOMContentLoaded', () => {
    const messagesContainer = document.getElementById('messages-container');
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const toggleSidebarBtn = document.getElementById('toggle-sidebar');
    const sidebar = document.getElementById('sidebar');
    const toggleThemeBtn = document.getElementById('toggle-theme');
    const companyNameEl = document.getElementById('company-name');
    const historyList = document.getElementById('history-list');
    const newChatBtn = document.getElementById('new-chat-btn');

    let currentChatId = null;

    // Fetch config from backend
    fetch('/api/config')
        .then(res => res.json())
        .then(config => {
            if (config.companyName) {
                companyNameEl.textContent = config.companyName;
                document.title = `${config.companyName} Chatbot`;
            }
            if (config.theme === 'light') {
                document.body.classList.remove('dark-theme');
            } else if (config.theme === 'dark') {
                document.body.classList.add('dark-theme');
            }
            if (!config.enableSidebar) {
                sidebar.classList.add('hidden');
                toggleSidebarBtn.style.display = 'none';
            }
        })
        .catch(err => console.error('Failed to load config:', err));

    // Toggle Sidebar
    toggleSidebarBtn.addEventListener('click', () => {
        sidebar.classList.toggle('hidden');
    });

    // Toggle Theme
    toggleThemeBtn.addEventListener('click', () => {
        document.body.classList.toggle('dark-theme');
    });

    function appendMessage(sender, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender === 'user' ? 'user-message' : 'ai-message'}`;
        
        const avatar = document.createElement('div');
        avatar.className = `avatar ${sender === 'ai' ? 'ai-avatar' : ''}`;
        avatar.textContent = sender === 'user' ? 'U' : 'AI';

        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = text;

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);

        messagesContainer.appendChild(msgDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function showTyping() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message ai-message typing';
        typingDiv.id = 'typing-indicator-msg';
        
        const avatar = document.createElement('div');
        avatar.className = 'avatar ai-avatar';
        avatar.textContent = 'AI';
        
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';

        typingDiv.appendChild(avatar);
        typingDiv.appendChild(indicator);
        
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function removeTyping() {
        const typingMsg = document.getElementById('typing-indicator-msg');
        if (typingMsg) {
            typingMsg.remove();
        }
    }

    // --- MULTI-CHAT LOGIC ---

    async function loadChats() {
        try {
            const res = await fetch('/api/chats');
            const data = await res.json();
            if (data.status === 'success') {
                historyList.innerHTML = '';
                if (data.chats.length === 0) {
                    await createNewChat();
                } else {
                    data.chats.forEach(chat => {
                        addChatToSidebar(chat.id, chat.title);
                    });
                    // Load the most recent chat
                    await loadChatHistory(data.chats[0].id);
                }
            }
        } catch (err) {
            console.error("Failed to load chats", err);
        }
    }

    function addChatToSidebar(id, title) {
        let li = document.getElementById(`chat-item-${id}`);
        if (!li) {
            li = document.createElement('li');
            li.id = `chat-item-${id}`;
            li.className = 'history-item';
            li.addEventListener('click', () => loadChatHistory(id));
            // insert at top
            historyList.insertBefore(li, historyList.firstChild);
        }
        li.textContent = title;
    }

    function setActiveChatInSidebar(id) {
        document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
        const activeItem = document.getElementById(`chat-item-${id}`);
        if (activeItem) activeItem.classList.add('active');
    }

    async function createNewChat() {
        try {
            const res = await fetch('/api/chats', { method: 'POST' });
            const data = await res.json();
            if (data.status === 'success') {
                addChatToSidebar(data.chat_id, data.title);
                await loadChatHistory(data.chat_id);
            }
        } catch (err) {
            console.error("Failed to create chat", err);
        }
    }

    newChatBtn.addEventListener('click', createNewChat);

    async function loadChatHistory(chatId) {
        try {
            currentChatId = chatId;
            setActiveChatInSidebar(chatId);
            messagesContainer.innerHTML = ''; // Clear current view
            
            const res = await fetch(`/api/chats/${chatId}`);
            const data = await res.json();
            if (data.status === 'success') {
                data.messages.forEach(msg => appendMessage(msg.sender, msg.text));
            }
        } catch (err) {
            console.error("Failed to load history", err);
        }
    }

    // Handle form submit with Real-time Stream Parsing
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = messageInput.value.trim();
        if (!text || !currentChatId) return;

        // Add User message
        appendMessage('user', text);
        messageInput.value = '';

        // Show typing indicator momentarily while waiting for first byte
        showTyping();

        try {
            // Hit the streaming route, passing the current chat ID for context!
            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: text, chat_id: currentChatId })
            });

            removeTyping();

            if (!response.ok) {
                appendMessage('ai', 'Error connecting to the AI backend.');
                return;
            }

            // Prepare dynamic bubble for stream
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message ai-message';
            
            const avatar = document.createElement('div');
            avatar.className = 'avatar ai-avatar';
            avatar.textContent = 'AI';

            const content = document.createElement('div');
            content.className = 'message-content';
            
            msgDiv.appendChild(avatar);
            msgDiv.appendChild(content);
            messagesContainer.appendChild(msgDiv);

            // Read the stream chunk by chunk
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let done = false;

            while (!done) {
                const { value, done: readerDone } = await reader.read();
                done = readerDone;
                
                if (value) {
                    const chunkStr = decoder.decode(value, { stream: true });
                    const lines = chunkStr.split('\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const dataStr = line.slice(6);
                            if (dataStr === '[DONE]') {
                                done = true;
                                break;
                            }
                            try {
                                const dataObj = JSON.parse(dataStr);
                                if (dataObj.error) {
                                    content.textContent += "\n[Error: " + dataObj.error + "]";
                                } else if (dataObj.chunk) {
                                    // Append text to the bubble in real-time
                                    content.textContent += dataObj.chunk;
                                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                                } else if (dataObj.title) {
                                    // Update auto-generated chat title in sidebar
                                    const li = document.getElementById(`chat-item-${currentChatId}`);
                                    if (li) li.textContent = dataObj.title;
                                }
                            } catch (err) {
                                console.error("Error parsing JSON chunk from stream", err);
                            }
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Chat error:', error);
            removeTyping();
            appendMessage('ai', 'Network error. Could not reach the server.');
        }
    });

    // Initialize application by loading chats
    loadChats();
});
