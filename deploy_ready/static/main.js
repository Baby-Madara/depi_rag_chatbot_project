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

    // -----------------------------------------------------------------------
    // Config
    // -----------------------------------------------------------------------
    fetch('/api/config')
        .then(res => res.json())
        .then(config => {
            if (config.companyName) {
                companyNameEl.textContent = config.companyName;
                document.title = `${config.companyName} Chatbot`;
            }
            if (config.theme === 'light') {
                document.body.classList.remove('dark-theme');
            } else {
                document.body.classList.add('dark-theme');
            }
            if (!config.enableSidebar) {
                sidebar.classList.add('hidden');
                toggleSidebarBtn.style.display = 'none';
            }
        })
        .catch(err => console.error('Failed to load config:', err));

    // -----------------------------------------------------------------------
    // Sidebar & theme toggles
    // -----------------------------------------------------------------------
    toggleSidebarBtn.addEventListener('click', () => sidebar.classList.toggle('hidden'));
    toggleThemeBtn.addEventListener('click', () => document.body.classList.toggle('dark-theme'));

    // -----------------------------------------------------------------------
    // Markdown renderer (NEW — from Sara)
    // Converts **bold** → <strong> and \n → <br> before display.
    // -----------------------------------------------------------------------
    function renderMarkdown(rawText) {
        let html = rawText
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')   // bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>')               // italic
            .replace(/\n/g, '<br>');                             // newlines
        return html;
    }

    // -----------------------------------------------------------------------
    // Typewriter effect (NEW — from Sara)
    // Streams HTML character-by-character without breaking mid-tag.
    // Only used for static (history) messages; live streaming uses the
    // chunk-appending path below so both effects work together.
    // -----------------------------------------------------------------------
    function typewriterAppend(element, htmlText, speed = 12) {
        let i = 0;
        function tick() {
            if (i >= htmlText.length) return;
            // Skip over entire HTML tags instantly so they don't render broken
            if (htmlText.charAt(i) === '<') {
                const tagEnd = htmlText.indexOf('>', i);
                if (tagEnd !== -1) { i = tagEnd + 1; }
            }
            element.innerHTML = htmlText.substring(0, i);
            i++;
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            setTimeout(tick, speed);
        }
        tick();
    }

    // -----------------------------------------------------------------------
    // Message helpers
    // -----------------------------------------------------------------------
    function _buildBubble(sender) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender === 'user' ? 'user-message' : 'ai-message'}`;

        const avatar = document.createElement('div');
        avatar.className = `avatar ${sender === 'ai' ? 'ai-avatar' : ''}`;
        avatar.textContent = sender === 'user' ? 'U' : 'AI';

        const content = document.createElement('div');
        content.className = 'message-content';

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);
        messagesContainer.appendChild(msgDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        return content; // caller gets a reference to write into
    }

    /**
     * appendMessage — used when loading history or showing a complete message.
     * AI messages get the typewriter effect; user messages appear instantly.
     */
    function appendMessage(sender, text) {
        const content = _buildBubble(sender);
        if (sender === 'ai') {
            // Render Markdown then typewrite
            typewriterAppend(content, renderMarkdown(text));
        } else {
            content.textContent = text;
        }
    }

    function showTyping() {
        const typingDiv = document.createElement('div');
        typingDiv.id = 'typing-indicator-msg';
        typingDiv.className = 'message ai-message typing';

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
        const el = document.getElementById('typing-indicator-msg');
        if (el) el.remove();
    }

    // -----------------------------------------------------------------------
    // Multi-chat logic
    // -----------------------------------------------------------------------
    async function loadChats() {
        try {
            const res = await fetch('/api/chats');
            const data = await res.json();
            if (data.status === 'success') {
                historyList.innerHTML = '';
                if (data.chats.length === 0) {
                    await createNewChat();
                } else {
                    data.chats.forEach(chat => addChatToSidebar(chat.id, chat.title));
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
            historyList.insertBefore(li, historyList.firstChild);
        }
        li.textContent = title;
    }

    function setActiveChatInSidebar(id) {
        document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
        const item = document.getElementById(`chat-item-${id}`);
        if (item) item.classList.add('active');
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
            messagesContainer.innerHTML = '';

            const res = await fetch(`/api/chats/${chatId}`);
            const data = await res.json();
            if (data.status === 'success') {
                data.messages.forEach(msg => appendMessage(msg.sender, msg.text));
            }
        } catch (err) {
            console.error("Failed to load history", err);
        }
    }

    // -----------------------------------------------------------------------
    // Send message — SSE streaming  +  live Markdown rendering
    // -----------------------------------------------------------------------
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = messageInput.value.trim();
        if (!text || !currentChatId) return;

        appendMessage('user', text);
        messageInput.value = '';
        showTyping();

        try {
            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, chat_id: currentChatId }),
            });

            removeTyping();

            if (!response.ok) {
                appendMessage('ai', 'Error connecting to the AI backend.');
                return;
            }

            // Live bubble that we append chunks into
            const content = _buildBubble('ai');

            // We accumulate the raw text and re-render Markdown on each chunk
            // so bold/italic markers resolve correctly even mid-stream.
            let rawAccumulated = '';

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let done = false;

            while (!done) {
                const { value, done: readerDone } = await reader.read();
                done = readerDone;

                if (value) {
                    const lines = decoder.decode(value, { stream: true }).split('\n');

                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        const dataStr = line.slice(6);

                        if (dataStr === '[DONE]') { done = true; break; }

                        try {
                            const obj = JSON.parse(dataStr);

                            if (obj.error) {
                                content.innerHTML += `<br><em style="color:red">[Error: ${obj.error}]</em>`;

                            } else if (obj.chunk) {
                                rawAccumulated += obj.chunk;
                                // Re-render Markdown on every chunk so markers resolve live
                                content.innerHTML = renderMarkdown(rawAccumulated);
                                messagesContainer.scrollTop = messagesContainer.scrollHeight;

                            } else if (obj.title) {
                                const li = document.getElementById(`chat-item-${currentChatId}`);
                                if (li) li.textContent = obj.title;
                            }
                        } catch (err) {
                            console.error('JSON parse error in stream chunk', err);
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

    // Boot
    loadChats();
});