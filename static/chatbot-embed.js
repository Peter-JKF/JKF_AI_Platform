(function() {
    // JKF Chatbot Embed Script
    // Configure before loading:
    //   window.CHATBOT_BASE_URL = 'https://your-jkf-app.domain';  (defaults to current origin)
    const CSS_URL = (window.CHATBOT_BASE_URL || window.location.origin) + '/static/chatbot-style.css';

    // Global variables
    let isAssistantResponding = false;
    let isToggleExpanded = false;
    let isExpansionScheduled = false;
    let userScrollPosition = 0;
    let shouldAutoScroll = true;
    let isFeedbackShown = false;
    let userHasInteracted = false;
    let threadId = null;
    let clientName = null;
    let companyId = null; // HMAC-signed company token from JKF Universe (null for anonymous)
    let hasAutoOpened = false;
    let leadFormEnabled = false;
    let leadFormFields = [];
    let showLeadForm = false;
    let leadFormSubmitted = false;
    let leadFormMinimized = false;
    let isActiveConversation = false;
    let currentLanguage = 'unknown';
    let currentCategory = 'Uncategorized';
    let knowledgeGaps = [];
    let csatShown = false;
    let csatTimer = null;
    let assistantMessageCount = 0;
    let csatRated = false; // Track if user has rated
    let csatHiddenByUser = false; // Track if CSAT was hidden by user activity
    let inactivityTimer = null; // Timer for 20-second inactivity detection

    // New variables for session management
    let sessionStartTime = null;
    const SESSION_DURATION = 30 * 60 * 1000; // 30 minutes in milliseconds
    let quickQuestions = []; // Will be loaded from client config

    // Load external CSS
    function loadCSS(url) {
        return new Promise((resolve, reject) => {
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = url;
            link.onload = resolve;
            link.onerror = reject;
            document.head.appendChild(link);
        });
    }

    // Create chatbot HTML structure
    function createChatbotHTML() {
        const chatbotHTML = `
            <div id="jkf-chatbot-container">
                <div id="jkf-chatbot-widget" class="jkf-chatbot-widget" style="display: none;">
                    <div class="jkf-chat-header">
                        <img src="" class="jkf-header-logo" alt="Company Logo">
                        <div class="jkf-status">
                            <div class="jkf-status-text"><strong><span id="jkf-chatbot-name">AI-assistent</span></strong></div>
                            <div class="jkf-status-active">
                                <div class="jkf-status-dot"></div>
                                <div class="jkf-status-text">Aktiv nu</div>
                            </div>
                        </div>
                        <div class="jkf-control-buttons">
                            <button id="jkf-restart-chat" class="jkf-round-arrow-btn"></button>
                            <button id="jkf-close-chat" class="jkf-close-btn">&times;</button>
                        </div>
                    </div>
                    <div id="jkf-disclaimer" class="jkf-disclaimer">
                        Dette er kun vejledende svar
                    </div>
                    <div id="jkf-chat-messages" class="jkf-chat-messages">
                        <div class="jkf-message-container jkf-assistant-message-container">
                            <div class="jkf-message jkf-assistant-message" id="jkf-welcome-message">Velkommen! Hvad kan jeg hjælpe med?</div>
                        </div>
                    </div>
                    <div class="jkf-warning-text">Opgiv ikke personlige oplysninger</div>
                    <div class="jkf-quick-questions-section">
                        <div class="jkf-quick-questions" id="jkf-quick-questions">
                            <!-- Quick questions will be inserted here -->
                        </div>
                    </div>
                    <div class="jkf-chat-input-section">
                        <div class="jkf-chat-input-wrapper">
                            <div class="jkf-chat-input">
                                <input type="text" id="jkf-user-input" placeholder="Stil et spørgsmål her...">
                                <div class="jkf-send-arrow"></div>
                            </div>
                        </div>
                        <div class="jkf-lead-form-button" style="display: none;">
                            <img src="https://jkf.dk/onewebmedia/demography_24dp_A0ABB6_FILL0_wght400_GRAD0_opsz24.svg" alt="Lead Form">
                        </div>
                    </div>

                    <div id="jkf-lead-form-container" class="jkf-lead-form-container" style="display: none;"></div>
                </div>
                <button id="jkf-chat-toggle" class="jkf-chat-toggle">
                    <div class="jkf-minimize-arrow"></div>
                </button>
            </div>
        `;
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = chatbotHTML;
        document.body.appendChild(tempDiv.firstElementChild);
    }

    function isDesktopDevice() {
        return window.innerWidth > 768 && !('ontouchstart' in window || navigator.maxTouchPoints > 0);
    }

    function getClientStorageKey(key) {
        return `${key}_${clientName}`;
    }
    
    // Render quick questions
    function renderQuickQuestions() {
        const quickQuestionsSection = document.querySelector('.jkf-quick-questions-section');
        const quickQuestionsContainer = document.getElementById('jkf-quick-questions');
        
        if (!quickQuestionsContainer || !quickQuestionsSection) {
            return;
        }
        
        quickQuestionsContainer.innerHTML = '';
        
        // Only render if quick questions exist and array is not empty
        if (!quickQuestions || quickQuestions.length === 0) {
            quickQuestionsSection.style.display = 'none';
            return;
        }
        
        // Check if there are existing messages beyond welcome message
        const chatMessages = document.getElementById('jkf-chat-messages');
        if (chatMessages) {
            const messageContainers = chatMessages.querySelectorAll('.jkf-message-container');
            if (messageContainers.length > 1) {
                quickQuestionsSection.style.display = 'none';
                return;
            }
        }
        
        quickQuestionsSection.style.display = 'block';
        
        quickQuestions.forEach(question => {
            const btn = document.createElement('button');
            btn.className = 'jkf-quick-question-btn';
            btn.textContent = question;
            btn.onclick = function() {
                if (!isAssistantResponding) {
                    document.getElementById('jkf-user-input').value = question;
                    sendMessage();
                }
            };
            quickQuestionsContainer.appendChild(btn);
        });
    }
    
    // Hide quick questions after first user interaction
    function hideQuickQuestions() {
        const quickQuestionsSection = document.querySelector('.jkf-quick-questions-section');
        if (quickQuestionsSection) {
            quickQuestionsSection.style.display = 'none';
        }
    }
    
    function saveChatState() {
        const chatState = {
            threadId: threadId,
            companyId: companyId,
            messages: document.getElementById('jkf-chat-messages').innerHTML,
            sessionStartTime: sessionStartTime,
            userHasInteracted: userHasInteracted,
            isActiveConversation: isActiveConversation,
            welcomeMessage: document.getElementById('jkf-welcome-message').textContent,
            currentLanguage: currentLanguage,
            currentCategory: currentCategory,
            knowledgeGaps: knowledgeGaps,
            csatShown: csatShown,
            assistantMessageCount: assistantMessageCount,
            csatRated: csatRated,
            leadFormMinimized: leadFormMinimized,
            leadFormSubmitted: leadFormSubmitted
        };
        localStorage.setItem(getClientStorageKey('jkfChatState'), JSON.stringify(chatState));
    }

    function loadChatState() {
        const chatState = JSON.parse(localStorage.getItem(getClientStorageKey('jkfChatState')));
        if (chatState) {
            const currentTime = new Date().getTime();
            if (currentTime - chatState.sessionStartTime < SESSION_DURATION) {
                threadId = chatState.threadId;
                // Restore companyId only if the window variable still matches (user still logged in)
                companyId = (window.CHATBOT_COMPANY_TOKEN || null) || chatState.companyId || null;
                document.getElementById('jkf-chat-messages').innerHTML = chatState.messages;
                sessionStartTime = chatState.sessionStartTime;
                userHasInteracted = chatState.userHasInteracted;
                isActiveConversation = chatState.isActiveConversation;
                if (chatState.welcomeMessage) {
                    document.getElementById('jkf-welcome-message').textContent = chatState.welcomeMessage;
                }
                currentLanguage = chatState.currentLanguage || 'unknown';
                currentCategory = chatState.currentCategory || 'Uncategorized';
                knowledgeGaps = chatState.knowledgeGaps || [];
                csatShown = chatState.csatShown || false;
                assistantMessageCount = chatState.assistantMessageCount || 0;
                csatRated = chatState.csatRated || false;
                leadFormMinimized = chatState.leadFormMinimized || false;
                leadFormSubmitted = chatState.leadFormSubmitted || false;
                updateMetadataDisplay();
                
                // Hide quick questions if user has interacted or there are multiple messages
                const chatMessages = document.getElementById('jkf-chat-messages');
                const messageContainers = chatMessages.querySelectorAll('.jkf-message-container');
                if (userHasInteracted || messageContainers.length > 1) {
                    hideQuickQuestions();
                }
                
                // If CSAT was shown but not rated, start inactivity timer
                if (csatShown && !csatRated && assistantMessageCount >= 1) {
                    startInactivityTimer();
                }
                
                return true;
            }
        }
        return false;
    }

    function clearChatState() {
        localStorage.removeItem(getClientStorageKey('jkfChatState'));
        localStorage.removeItem(getClientStorageKey('jkfCSATShown'));
        isActiveConversation = false;
        csatShown = false;
        assistantMessageCount = 0;
        csatRated = false;
        csatHiddenByUser = false;
        leadFormMinimized = false;
        leadFormSubmitted = false;
        showLeadForm = false;
        if (csatTimer) {
            clearTimeout(csatTimer);
            csatTimer = null;
        }
        if (inactivityTimer) {
            clearTimeout(inactivityTimer);
            inactivityTimer = null;
        }
    }

    function isUserAtBottom() {
        const chatMessages = document.getElementById('jkf-chat-messages');
        return chatMessages.scrollHeight - chatMessages.clientHeight <= chatMessages.scrollTop + 5;
    }

    function scrollToBottom() {
        const chatMessages = document.getElementById('jkf-chat-messages');
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function sendMessage() {
        if (isAssistantResponding) return;
        if (!threadId) return; // waiting for new thread_id after reset

        const userInput = document.getElementById('jkf-user-input');
        const message = userInput.value.trim();

        if (message) {
            // Hide quick questions after first user interaction
            hideQuickQuestions();
            
            // Hide CSAT if visible and user is sending a message (they're still engaged)
            hideCSATIfVisible();
            
            // Reset inactivity timer
            resetInactivityTimer();
            
            userHasInteracted = true;
            isActiveConversation = true;
            shouldAutoScroll = isUserAtBottom();
            addMessageToChat('You', message);
            userInput.value = '';
    
            isAssistantResponding = true;
            disableUserInput();
    
            const typingIndicator = addTypingIndicator();
    
            const baseUrl = window.CHATBOT_BASE_URL || window.location.origin;
    
            // Use streaming endpoint
            fetch(`${baseUrl}/chat_stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message, thread_id: threadId, client_name: clientName, company_id: companyId }),
            })
            .then(response => {
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                // Keep typing indicator until first content arrives
                const chatMessages = document.getElementById('jkf-chat-messages');
                let assistantMessageDiv = null;
                let firstContentReceived = false;
                
                function readStream() {
                    reader.read().then(({ done, value }) => {
                        if (done) {
                            // Clean up typing indicator if still present
                            if (typingIndicator && typingIndicator.parentNode) {
                                chatMessages.removeChild(typingIndicator);
                            }
                            if (!assistantMessageDiv) {
                                // Stream closed before any content arrived (e.g. connection
                                // timeout during function call). Show a friendly error.
                                assistantMessageDiv = addMessageToChat('Assistant', '');
                                assistantMessageDiv.innerHTML = 'Beklager, der opstod en fejl. Prøv venligst igen.';
                            } else if (!assistantMessageDiv.innerHTML.trim()) {
                                // Message div exists but is empty
                                assistantMessageDiv.innerHTML = 'Beklager, der opstod en fejl. Prøv venligst igen.';
                            }
                            isAssistantResponding = false;
                            enableUserInput();
                            saveChatState();
                            return;
                        }
                        
                        const chunk = decoder.decode(value, { stream: true });
                        const lines = chunk.split('\n');
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.substring(6));
                                    
                                    if (data.type === 'thread_id') {
                                        threadId = data.content;
                                    } else if (data.type === 'tool_call') {
                                        // Show a status label while a BC tool runs
                                        if (!assistantMessageDiv) {
                                            if (typingIndicator && typingIndicator.parentNode) {
                                                chatMessages.removeChild(typingIndicator);
                                            }
                                            assistantMessageDiv = addMessageToChat('Assistant', '');
                                        }
                                        assistantMessageDiv.innerHTML =
                                            `<div class="jkf-tool-status">` +
                                            `<div class="jkf-tool-status-dots"><span></span><span></span><span></span></div>` +
                                            `<span class="jkf-tool-status-label">${data.label || 'Arbejder…'}</span>` +
                                            `</div>`;
                                        if (shouldAutoScroll) scrollToBottom();
                                    } else if (data.type === 'replace' || data.type === 'content') {
                                        // On first content, remove typing indicator and create message div
                                        if (!firstContentReceived) {
                                            firstContentReceived = true;
                                            if (typingIndicator && typingIndicator.parentNode) {
                                                chatMessages.removeChild(typingIndicator);
                                            }
                                            if (!assistantMessageDiv) {
                                                assistantMessageDiv = addMessageToChat('Assistant', '');
                                            }
                                        }
                                        
                                        // Strip [METADATA] blocks on frontend as safety net
                                        let displayContent = data.content;
                                        if (displayContent.includes('[METADATA]')) {
                                            displayContent = displayContent.replace(/\[METADATA\][\s\S]*?\[\/METADATA\]/g, '');
                                        }
                                        
                                        // Also strip any partial metadata tag at the end (extra safety)
                                        // This prevents "[METADATA" without closing "]" from appearing
                                        const metadataStart = '[METADATA]';
                                        for (let i = 1; i < metadataStart.length; i++) {
                                            if (displayContent.endsWith(metadataStart.substring(0, i))) {
                                                displayContent = displayContent.substring(0, displayContent.length - i);
                                                break;
                                            }
                                        }
                                        
                                        // Replace entire content with formatted text
                                        assistantMessageDiv.innerHTML = renderMarkdown(displayContent);
                                        if (shouldAutoScroll) {
                                            scrollToBottom();
                                        }
                                    } else if (data.type === 'metadata') {
                                        // Update metadata
                                        if (data.content) {
                                            currentLanguage = data.content.language || currentLanguage;
                                            currentCategory = data.content.category || currentCategory;
                                            if (data.content.knowledge_gaps) {
                                                knowledgeGaps = Array.isArray(knowledgeGaps) ? knowledgeGaps : [];
                                                knowledgeGaps.push(data.content.knowledge_gaps);
                                            }
                                            updateMetadataDisplay();
                                        }
                                    } else if (data.type === 'done') {
                                        threadId = data.thread_id || threadId;
                                        saveChatState();
                                        
                                        if (leadFormEnabled && !showLeadForm && !leadFormSubmitted && !leadFormMinimized) {
                                            showLeadForm = true; // Set immediately to block duplicate timers
                                            setTimeout(() => {
                                                renderLeadForm();
                                            }, 30000);
                                        }
                                        
                                        isAssistantResponding = false;
                                        enableUserInput();
                                        
                                        // Schedule CSAT after each assistant message (function guards threshold internally)
                                        scheduleCSAT();
                                        if (csatShown && !csatRated) {
                                            // If CSAT was already shown but not rated, restart inactivity timer
                                            resetInactivityTimer();
                                        }
                                        return;
                                    } else if (data.type === 'error') {
                                        console.error('Streaming error:', data.content);
                                        if (typingIndicator && typingIndicator.parentNode) {
                                            chatMessages.removeChild(typingIndicator);
                                        }
                                        if (!assistantMessageDiv) {
                                            assistantMessageDiv = addMessageToChat('Assistant', '');
                                        }
                                        assistantMessageDiv.innerHTML = 'Der opstod en fejl. Prøv venligst igen.';
                                        isAssistantResponding = false;
                                        enableUserInput();
                                        return;
                                    }
                                } catch (e) {
                                    console.error('Error parsing SSE data:', e);
                                }
                            }
                        }
                        
                        readStream();
                    }).catch(error => {
                        console.error('Error reading stream:', error);
                        isAssistantResponding = false;
                        enableUserInput();
                        saveChatState();
                    });
                }
                
                readStream();
            })
            .catch(error => {
                console.error('Error:', error);
                const chatMessages = document.getElementById('jkf-chat-messages');
                if (typingIndicator && typingIndicator.parentNode) {
                    chatMessages.removeChild(typingIndicator);
                }
                addMessageToChat('Assistant', 'Der opstod en fejl. Prøv venligst igen.');
                isAssistantResponding = false;
                enableUserInput();
                saveChatState();
            });
        }
    }
    
    function updateMetadataDisplay() {
        // Metadata is tracked in variables for analytics
    }

    function validateField(name, value) {
        switch (name) {
            case 'postnummer':
                return {
                    isValid: /^\d{4}$/.test(value),
                    errorMessage: 'Postnummeret skal være præcis 4 cifre'
                };
            case 'Telefon':
                return {
                    isValid: /^\d{8}$/.test(value),
                    errorMessage: 'Telefonnummeret skal være præcis 8 cifre'
                };
            case 'Email':
                // More permissive regex that allows Danish characters
                return {
                    isValid: /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value) && value.includes('@'),
                    errorMessage: 'Indtast venligst en gyldig e-mailadresse'
                };
            default:
                return { isValid: true, errorMessage: '' };
        }
    }
    
    function renderLeadForm() {
        const config = window.jkfConfig;
        
        if (!leadFormEnabled || leadFormFields.length === 0) {
            return;
        }
    
        const leadFormContainer = document.getElementById('jkf-lead-form-container');
        leadFormContainer.style.display = 'flex';
        
        let formFieldsHtml = '';
        leadFormFields.forEach(field => {
            const defaultValue = field.defaultValue || '';
            const fieldId = `${field.name}-field`;
            
            switch(field.type) {
                case 'text':
                case 'email':
                case 'tel':
                    formFieldsHtml += `
                        <div class="jkf-form-group">
                            <label for="${fieldId}">${field.label}</label>
                            <input type="${field.type}" 
                                   id="${fieldId}" 
                                   name="${field.name}" 
                                   placeholder="${field.placeholder || ''}"
                                   value="${defaultValue}"
                                   ${field.required ? 'required' : ''}>
                            <div id="${fieldId}-error" class="jkf-error-message" style="display: none; color: red; font-size: 12px; margin-top: 4px;"></div>
                        </div>`;
                    break;
                    
                case 'calendar':
                    const defaultDate = field.defaultValue?.date || '';
                    const defaultTime = field.defaultValue?.time || '';
                    formFieldsHtml += `
                        <div class="jkf-form-group">
                            <label for="${field.name}">${field.label}</label>
                            <input type="date" 
                                   id="${field.name}_date" 
                                   name="${field.name}_date"
                                   value="${defaultDate}"
                                   ${field.required ? 'required' : ''}>
                            ${field.includeTime ? `
                                <input type="time" 
                                       id="${field.name}_time" 
                                       name="${field.name}_time"
                                       value="${defaultTime}"
                                       ${field.required ? 'required' : ''}>
                            ` : ''}
                        </div>`;
                    break;
                    
                case 'dropdown':
                    formFieldsHtml += `
                        <div class="jkf-form-group">
                            <label for="${field.name}">${field.label}</label>
                            <select id="${field.name}" 
                                    name="${field.name}"
                                    ${field.required ? 'required' : ''}>
                                <option value="">Vælg...</option>
                                ${field.options.map(opt => `
                                    <option value="${opt.value}" ${opt.value === defaultValue ? 'selected' : ''}>
                                        ${opt.label}
                                    </option>
                                `).join('')}
                            </select>
                        </div>`;
                    break;
            }
        });

        // Add privacy policy checkbox before the submit button
        let privacyHtml = '';
        if (config.privacy_consent_enabled) {
            privacyHtml = `
                <div class="jkf-privacy-checkbox">
                    <div class="jkf-checkbox-wrapper">
                        <input type="checkbox" 
                               id="privacy_consent" 
                               name="privacy_consent" 
                               required>
                        <label for="privacy_consent" class="jkf-checkbox-label">
                            ${config.privacy_consent_text} <a href="${config.privacy_policy_url}" 
                               target="_blank" 
                               class="privacy-link">${config.privacy_policy_link_text}</a>
                        </label>
                    </div>
                </div>
            `;
        }

    // Add privacy checkbox CSS to your existing CSS file
    const style = document.createElement('style');
    style.textContent = `
        .jkf-privacy-checkbox {
            margin: 15px 0 !important;
        }
    
        .jkf-checkbox-wrapper {
            display: flex !important;
            align-items: center !important;
            gap: 4px !important;
            margin-top: 5px !important;
        }
    
        .jkf-privacy-checkbox input[type="checkbox"] {
            margin: 0 !important;
            cursor: pointer !important;
            width: 16px !important;
            height: 16px !important;
            flex-shrink: 0 !important;
        }
    
        .jkf-checkbox-label {
            font-size: 12px !important;
            color: var(--jkf-secondary-color) !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
            gap: 4px !important;
            margin: 0 !important;
        }
    
        .jkf-privacy-checkbox .privacy-link {
            color: var(--jkf-primary-color) !important;
            text-decoration: none !important;
            font-size: 12px !important;
        }
    
        .jkf-privacy-checkbox .privacy-link:hover {
            text-decoration: underline !important;
        }
    `;
    document.head.appendChild(style);
    
        // Add newsletter signup if enabled
        const newsletterHtml = newsletterSignup ? `
            <div class="newsletter-checkbox">
                <input type="checkbox" id="newsletter_signup" name="newsletter_signup">
                <label for="newsletter_signup">Ja tak til nyhedsbrev</label>
            </div>
        ` : '';
    
        leadFormContainer.innerHTML = `
            <div class="jkf-lead-form-backdrop"></div>
            <div class="jkf-lead-form">
                <div class="jkf-lead-form-header">
                    <h3>${leadFormTitle}</h3>
                    <button class="jkf-close-lead-form">&times;</button>
                </div>
                ${leadFormDescription ? `<p class="jkf-lead-form-description">${leadFormDescription}</p>` : ''}
                <form id="jkf-lead-capture-form">
                    ${formFieldsHtml}
                    ${privacyHtml}
                    ${newsletterHtml}
                </form>
                <div class="jkf-lead-form-submit">
                    <button type="submit" form="jkf-lead-capture-form">${leadFormButtonText}</button>
                </div>
            </div>
        `;

        leadFormFields.forEach(field => {
            const inputElement = document.getElementById(`${field.name}-field`);
            if (inputElement) {
                // Add validation on input
                inputElement.addEventListener('input', function() {
                    const validation = validateField(field.name, this.value);
                    const errorElement = document.getElementById(`${field.name}-field-error`);
                    if (errorElement) {
                        if (!validation.isValid && this.value !== '') {  // Only show error if field isn't empty
                            errorElement.textContent = validation.errorMessage;
                            errorElement.style.display = 'block';
                            this.style.borderColor = 'red';
                        } else {
                            errorElement.style.display = 'none';
                            this.style.borderColor = '';
                        }
                    }
                });
            }
        });
    
        // Add event listeners
        document.querySelector('.jkf-close-lead-form').addEventListener('click', toggleLeadForm);
        document.getElementById('jkf-lead-capture-form').addEventListener('submit', handleLeadFormSubmit);
        document.querySelector('.jkf-lead-form-backdrop').addEventListener('click', toggleLeadForm);
    }
    
    function toggleLeadForm() {
        const leadFormContainer = document.getElementById('jkf-lead-form-container');
        if (leadFormContainer.style.display === 'none' || leadFormContainer.style.display === '') {
            renderLeadForm();
        } else {
            leadFormContainer.style.display = 'none';
            leadFormMinimized = true; // Mark as closed so it won't show again
            saveChatState(); // Save state to persist leadFormMinimized
        }
    }
    
    function handleLeadFormSubmit(e) {
        e.preventDefault();
        const form = e.target;
        const formData = new FormData(form);
        const config = window.jkfConfig;
        
        // Add validation check here
        let isValid = true;
        leadFormFields.forEach(field => {
            const value = formData.get(field.name);
            const validation = validateField(field.name, value);
            if (!validation.isValid) {
                isValid = false;
                // Show error message - using the correct IDs that match your form
                const errorElement = document.getElementById(`${field.name}-field-error`);
                if (errorElement) {
                    errorElement.textContent = validation.errorMessage;
                    errorElement.style.display = 'block';
                }
                // Add red border to invalid field - using the correct IDs that match your form
                const inputElement = document.getElementById(`${field.name}-field`);
                if (inputElement) {
                    inputElement.style.borderColor = 'red';
                }
            }
        });
    
        if (!isValid) {
            return;
        }
        
        let leadData = {
            client_name: clientName,
            thread_id: threadId
        };

        // Check privacy consent if enabled
        if (config.privacy_consent_enabled) {
            const privacyConsent = formData.get('privacy_consent') === 'on';
            if (!privacyConsent) {
                alert('Du skal acceptere privatlivspolitikken for at fortsætte');
                return;
            }
            leadData.privacy_consent = privacyConsent;
        }
    
        // Only include newsletter_signup if newsletter signup is enabled
        if (newsletterSignup) {
            leadData.newsletter_signup = formData.get('newsletter_signup') === 'on';
        }
    
        // Process each form field based on configuration and maintain order
        leadFormFields.forEach(field => {
            const fieldName = field.name;
            const fieldType = field.type;
            
            if (fieldType === 'dropdown') {
                const value = formData.get(fieldName);
                if (value) {
                    let optionLabel = value;
        
                    if (Array.isArray(field.options)) {
                        // Check if options are grouped
                        if (field.options.some(opt => opt.options)) {
                            // Options are grouped
                            const group = field.options.find(g => g.options.find(opt => opt.value === value));
                            const option = group ? group.options.find(opt => opt.value === value) : null;
                            optionLabel = option ? option.label : value;
                        } else {
                            // Options are not grouped
                            const option = field.options.find(opt => opt.value === value);
                            optionLabel = option ? option.label : value;
                        }
                    }
        
                    // Store the value with both the specific field name and as dropdown_value for routing
                    leadData[fieldName] = value;
                    leadData[`${fieldName}_label`] = optionLabel;
                    leadData.dropdown_value = value;  // Add this for routing
                }
            } else if (fieldType === 'calendar') {
                const dateValue = formData.get(`${fieldName}_date`);
                const timeValue = formData.get(`${fieldName}_time`);
                if (dateValue) {
                    leadData[`${fieldName}_date`] = dateValue;
                }
                if (timeValue) {
                    leadData[`${fieldName}_time`] = timeValue;
                }
            } else {
                const value = formData.get(fieldName);
                if (value) {
                    leadData[fieldName] = value;
                }
            }
        });
    
        const baseUrl = window.CHATBOT_BASE_URL || window.location.origin;
        
        fetch(`${baseUrl}/submit_lead`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(leadData),
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                leadFormThankYouTitle = data.thank_you_title;
                leadFormThankYouMessage = data.thank_you_message;
                showThankYouMessage();
                saveChatState(); // Save state to persist leadFormSubmitted
            } else {
                console.error('Error submitting lead:', data.message);
                alert('Der opstod en fejl. Prøv venligst igen.');
            }
        })
        .catch(error => {
            console.error('Error submitting lead:', error);
            alert('Der opstod en fejl. Prøv venligst igen.');
        });
    }
    
    function showThankYouMessage() {
        leadFormSubmitted = true; // Mark as submitted so it won't show again
        const leadFormContainer = document.getElementById('jkf-lead-form-container');
        leadFormContainer.innerHTML = `
            <div class="jkf-lead-form-backdrop"></div>
            <div class="jkf-lead-form thank-you-message">
                <button class="jkf-close-thank-you">Luk</button>
                <img src="${document.querySelector('.jkf-header-logo').src}" alt="Company Logo" class="jkf-thank-you-logo">
                <h3>${leadFormThankYouTitle}</h3>
                <p>${leadFormThankYouMessage}</p>
            </div>
        `;
    
        leadFormContainer.style.display = 'flex';
        document.querySelector('.jkf-close-thank-you').addEventListener('click', closeLeadForm);
        document.querySelector('.jkf-lead-form-backdrop').addEventListener('click', closeLeadForm);
    }
    
    function closeLeadForm() {
        const leadFormContainer = document.getElementById('jkf-lead-form-container');
        leadFormContainer.style.display = 'none';
        leadFormContainer.innerHTML = '';
    }

    function disableUserInput() {
        const sendArrow = document.querySelector('.jkf-send-arrow');
        sendArrow.classList.add('disabled');
    }

    function enableUserInput() {
        const sendArrow = document.querySelector('.jkf-send-arrow');
        sendArrow.classList.remove('disabled');
    }

    function initChat() {
        if (!loadChatState()) {
            const baseUrl = window.CHATBOT_BASE_URL || window.location.origin;

            fetch(`${baseUrl}/init_thread`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            })
            .then(response => response.json())
            .then(data => {
                threadId = data.thread_id;
                sessionStartTime = new Date().getTime();
                saveChatState();
            })
            .catch(error => console.error('Error initializing thread:', error));
        }
    }

    function trackUrlClick(url) {
        const baseUrl = window.CHATBOT_BASE_URL || window.location.origin;
        fetch(`${baseUrl}/track_click`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ thread_id: threadId, url: url }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.status !== 'success') {
                console.error('Error tracking URL click:', data.error);
            }
        })
        .catch(error => console.error('Error:', error));
    }

    function addUrlClickListeners() {
        const chatMessages = document.getElementById('jkf-chat-messages');
        chatMessages.addEventListener('click', function(event) {
            if (event.target.tagName === 'A') {
                event.preventDefault();
                const url = event.target.href;
                trackUrlClick(url);
                window.open(url, '_blank');
            }
        });
    }

    function addTypingIndicator() {
        const chatMessages = document.getElementById('jkf-chat-messages');
        const typingContainer = document.createElement('div');
        typingContainer.className = 'jkf-message-container jkf-assistant-message-container';
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'jkf-message jkf-assistant-message jkf-typing-indicator';
        typingIndicator.innerHTML = '<span></span><span></span><span></span>';
        typingContainer.appendChild(typingIndicator);
        chatMessages.appendChild(typingContainer);
        if (shouldAutoScroll) {
            scrollToBottom();
        }
        return typingContainer;
    }

    // Render basic Markdown to HTML (links, bold, italic, line breaks)
    function renderMarkdown(text) {
        if (!text) return '';
        // Markdown links [text](url) → clickable <a>
        text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
            '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
        // **bold**
        text = text.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
        // *italic* (not adjacent to another *)
        text = text.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, '<em>$1</em>');
        // Newlines → <br>
        text = text.replace(/\n/g, '<br>');
        return text;
    }

    function addMessageToChat(sender, messageContent) {
        const chatMessages = document.getElementById('jkf-chat-messages');
        const messageContainer = document.createElement('div');
        messageContainer.className = 'jkf-message-container';
        const messageDiv = document.createElement('div');
        messageDiv.className = 'jkf-message';
    
        if (sender.toLowerCase() === 'you') {
            messageContainer.classList.add('jkf-user-message-container');
            messageDiv.classList.add('jkf-user-message');
            messageDiv.textContent = messageContent;
        } else {
            messageContainer.classList.add('jkf-assistant-message-container');
            messageDiv.classList.add('jkf-assistant-message');
            messageDiv.innerHTML = renderMarkdown(messageContent);  // Render markdown + HTML formatting
        }
    
        messageContainer.appendChild(messageDiv);
        chatMessages.appendChild(messageContainer);
    
        if (shouldAutoScroll) {
            scrollToBottom();
        }
        saveChatState();
        return messageDiv;
    }

    // Typing animation removed - now using real-time streaming
    
    function scheduleCSAT() {
        // Only show CSAT once per session
        if (csatShown || csatRated) return;
        
        assistantMessageCount++;
        
        // Clear any existing timer
        if (csatTimer) {
            clearTimeout(csatTimer);
        }
        
        // Require at least 2 assistant responses before showing CSAT
        if (assistantMessageCount < 2) return;
        
        // Schedule CSAT to appear after 10 seconds of inactivity.
        // If the assistant is still responding (tool call in progress), reschedule.
        csatTimer = setTimeout(() => {
            if (!csatShown && !csatRated && assistantMessageCount >= 2) {
                if (isAssistantResponding) {
                    scheduleCSAT();
                } else {
                    showCSATInline();
                }
            }
        }, 10000);
    }
    
    function showCSATInline(customText) {
        if (csatShown || csatRated) return;
        
        csatShown = true;
        
        // Use custom text or default text
        const questionText = customText || 'Fandt du det svar, du ledte efter?';
        
        // Create CSAT container with inline styles matching chatbot's clean design
        const csatContainer = document.createElement('div');
        csatContainer.className = 'jkf-csat-container';
        csatContainer.id = 'jkf-csat-inline';
        csatContainer.style.cssText = `
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 18px 16px !important;
            margin: 12px 16px !important;
            background: #f5f5f5 !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
            max-width: calc(100% - 32px) !important;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif !important;
            box-sizing: border-box !important;
        `;
        
        csatContainer.innerHTML = `
            <div class="jkf-csat-question" style="
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif !important;
                font-size: 14px !important;
                font-weight: 500 !important;
                color: #000000 !important;
                margin: 0 0 14px 0 !important;
                text-align: center !important;
                line-height: 1.4 !important;
            ">${questionText}</div>
            <div class="jkf-csat-emojis" style="
                display: flex !important;
                gap: 6px !important;
                justify-content: center !important;
                align-items: flex-start !important;
                margin: 0 !important;
            ">
                <div class="jkf-csat-emoji-wrapper" style="display: flex !important; flex-direction: column !important; align-items: center !important; gap: 4px !important;">
                    <button class="jkf-csat-emoji" data-rating="1" style="
                        background: transparent !important;
                        border: none !important;
                        border-radius: 8px !important;
                        width: 46px !important;
                        height: 46px !important;
                        font-size: 32px !important;
                        cursor: pointer !important;
                        display: flex !important;
                        align-items: center !important;
                        justify-content: center !important;
                        margin: 0 !important;
                        padding: 0 !important;
                        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
                        opacity: 0.7 !important;
                    "><img src="https://cdn.jsdelivr.net/npm/emoji-datasource-apple/img/apple/64/1f620.png" style="width:32px !important;height:32px !important;pointer-events:none !important;display:block !important;" /></button>
                    <span class="jkf-csat-label" style="
                        font-size: 11px !important;
                        color: #333333 !important;
                        opacity: 0 !important;
                        transition: opacity 0.2s ease !important;
                        text-align: center !important;
                        white-space: nowrap !important;
                        pointer-events: none !important;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif !important;
                    ">Forfærdelig</span>
                </div>
                <div class="jkf-csat-emoji-wrapper" style="display: flex !important; flex-direction: column !important; align-items: center !important; gap: 4px !important;">
                    <button class="jkf-csat-emoji" data-rating="2" style="
                        background: transparent !important;
                        border: none !important;
                        border-radius: 8px !important;
                        width: 46px !important;
                        height: 46px !important;
                        font-size: 32px !important;
                        cursor: pointer !important;
                        display: flex !important;
                        align-items: center !important;
                        justify-content: center !important;
                        margin: 0 !important;
                        padding: 0 !important;
                        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
                        opacity: 0.7 !important;
                    "><img src="https://cdn.jsdelivr.net/npm/emoji-datasource-apple/img/apple/64/2639-fe0f.png" style="width:32px !important;height:32px !important;pointer-events:none !important;display:block !important;" /></button>
                    <span class="jkf-csat-label" style="
                        font-size: 11px !important;
                        color: #333333 !important;
                        opacity: 0 !important;
                        transition: opacity 0.2s ease !important;
                        text-align: center !important;
                        white-space: nowrap !important;
                        pointer-events: none !important;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif !important;
                    ">Dårlig</span>
                </div>
                <div class="jkf-csat-emoji-wrapper" style="display: flex !important; flex-direction: column !important; align-items: center !important; gap: 4px !important;">
                    <button class="jkf-csat-emoji" data-rating="3" style="
                        background: transparent !important;
                        border: none !important;
                        border-radius: 8px !important;
                        width: 46px !important;
                        height: 46px !important;
                        font-size: 32px !important;
                        cursor: pointer !important;
                        display: flex !important;
                        align-items: center !important;
                        justify-content: center !important;
                        margin: 0 !important;
                        padding: 0 !important;
                        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
                        opacity: 0.7 !important;
                    "><img src="https://cdn.jsdelivr.net/npm/emoji-datasource-apple/img/apple/64/1f610.png" style="width:32px !important;height:32px !important;pointer-events:none !important;display:block !important;" /></button>
                    <span class="jkf-csat-label" style="
                        font-size: 11px !important;
                        color: #333333 !important;
                        opacity: 0 !important;
                        transition: opacity 0.2s ease !important;
                        text-align: center !important;
                        white-space: nowrap !important;
                        pointer-events: none !important;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif !important;
                    ">Okay</span>
                </div>
                <div class="jkf-csat-emoji-wrapper" style="display: flex !important; flex-direction: column !important; align-items: center !important; gap: 4px !important;">
                    <button class="jkf-csat-emoji" data-rating="4" style="
                        background: transparent !important;
                        border: none !important;
                        border-radius: 8px !important;
                        width: 46px !important;
                        height: 46px !important;
                        font-size: 32px !important;
                        cursor: pointer !important;
                        display: flex !important;
                        align-items: center !important;
                        justify-content: center !important;
                        margin: 0 !important;
                        padding: 0 !important;
                        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
                        opacity: 0.7 !important;
                    "><img src="https://cdn.jsdelivr.net/npm/emoji-datasource-apple/img/apple/64/1f603.png" style="width:32px !important;height:32px !important;pointer-events:none !important;display:block !important;" /></button>
                    <span class="jkf-csat-label" style="
                        font-size: 11px !important;
                        color: #333333 !important;
                        opacity: 0 !important;
                        transition: opacity 0.2s ease !important;
                        text-align: center !important;
                        white-space: nowrap !important;
                        pointer-events: none !important;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif !important;
                    ">God</span>
                </div>
                <div class="jkf-csat-emoji-wrapper" style="display: flex !important; flex-direction: column !important; align-items: center !important; gap: 4px !important;">
                    <button class="jkf-csat-emoji" data-rating="5" style="
                        background: transparent !important;
                        border: none !important;
                        border-radius: 8px !important;
                        width: 46px !important;
                        height: 46px !important;
                        font-size: 32px !important;
                        cursor: pointer !important;
                        display: flex !important;
                        align-items: center !important;
                        justify-content: center !important;
                        margin: 0 !important;
                        padding: 0 !important;
                        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
                        opacity: 0.7 !important;
                    "><img src="https://cdn.jsdelivr.net/npm/emoji-datasource-apple/img/apple/64/1f929.png" style="width:32px !important;height:32px !important;pointer-events:none !important;display:block !important;" /></button>
                    <span class="jkf-csat-label" style="
                        font-size: 11px !important;
                        color: #333333 !important;
                        opacity: 0 !important;
                        transition: opacity 0.2s ease !important;
                        text-align: center !important;
                        white-space: nowrap !important;
                        pointer-events: none !important;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif !important;
                    ">Fantastisk</span>
                </div>
            </div>
        `;
        
        // Add to chat messages first
        const chatMessages = document.getElementById('jkf-chat-messages');
        chatMessages.appendChild(csatContainer);
        
        // Scroll to show CSAT
        scrollToBottom();
        
        // Add hover and click handlers (combined to avoid duplicate declaration)
        const emojiWrappers = csatContainer.querySelectorAll('.jkf-csat-emoji-wrapper');
        emojiWrappers.forEach(wrapper => {
            const button = wrapper.querySelector('.jkf-csat-emoji');
            const label = wrapper.querySelector('.jkf-csat-label');
            
            // Hover effects - no background box, just scale and opacity
            button.addEventListener('mouseenter', function() {
                this.style.transform = 'scale(1.2)';
                this.style.opacity = '1';
                if (label) label.style.opacity = '1';
            });
            button.addEventListener('mouseleave', function() {
                this.style.transform = 'scale(1)';
                this.style.opacity = '0.7';
                if (label) label.style.opacity = '0';
            });
            
            // Click handler
            button.addEventListener('click', function(e) {
                e.stopPropagation();
                const rating = parseInt(this.dataset.rating);
                submitCSATFeedback(rating, csatContainer);
            });
        });
        
        // Save state
        localStorage.setItem(getClientStorageKey('jkfCSATShown'), 'true');
        
        // Start inactivity timer for re-showing CSAT
        startInactivityTimer();
    }
    
    function hideCSATIfVisible() {
        // Remove inline CSAT if it exists
        const csatContainer = document.getElementById('jkf-csat-inline');
        if (csatContainer) {
            if (!csatRated) {
                csatContainer.remove(); // Remove from DOM instead of just hiding
                csatHiddenByUser = true;
            }
        }
        
    }
    
    function startInactivityTimer() {
        // Clear existing timer
        if (inactivityTimer) {
            clearTimeout(inactivityTimer);
        }
        
        // Don't start timer if CSAT already rated
        if (csatRated) {
            return;
        }
        
        // Start 20-second inactivity timer
        inactivityTimer = setTimeout(() => {
            if (csatHiddenByUser && !csatRated) {
                reShowCSAT();
            }
        }, 20000); // 20 seconds
    }
    
    function resetInactivityTimer() {
        // User is active, reset the timer
        if (inactivityTimer) {
            clearTimeout(inactivityTimer);
            inactivityTimer = null;
        }
        startInactivityTimer();
    }
    
    function reShowCSAT() {
        if (csatRated) return;
        
        const csatContainer = document.getElementById('jkf-csat-inline');
        const chatMessages = document.getElementById('jkf-chat-messages');
        
        if (csatContainer) {
            // CSAT element exists - move it to bottom of chat and make visible
            csatContainer.style.display = 'flex';
            csatHiddenByUser = false;
            
            // Remove from current position and append to bottom
            if (csatContainer.parentNode) {
                csatContainer.parentNode.removeChild(csatContainer);
            }
            chatMessages.appendChild(csatContainer);
            
            scrollToBottom();
        } else {
            // CSAT element doesn't exist (was removed when user sent message)
            // Create it again
            csatShown = false; // Reset so showCSATInline can work
            csatHiddenByUser = false; // Reset the flag
            showCSATInline();
        }
    }
    
    function submitCSATFeedback(rating, csatContainer) {
        const baseUrl = window.CHATBOT_BASE_URL || window.location.origin;
        
        // Mark as rated and stop inactivity timer
        csatRated = true;
        if (inactivityTimer) {
            clearTimeout(inactivityTimer);
            inactivityTimer = null;
        }
        
        fetch(`${baseUrl}/submit_feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ rating: rating, thread_id: threadId }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Save state to persist csatRated
                saveChatState();
                
                // Pick thank-you message based on rating
                let thankYouText;
                if (rating >= 4) {
                    thankYouText = 'Fantastisk! Vi er glade for at have hjulpet dig 😊';
                } else if (rating === 3) {
                    thankYouText = 'Tak for din feedback — vi arbejder løbende på at blive bedre.';
                } else {
                    thankYouText = 'Vi er kede af, at vi ikke kunne hjælpe. Vi arbejder på at gøre det bedre næste gang.';
                }
                
                // Replace CSAT with thank-you message
                csatContainer.innerHTML = `
                    <div class="jkf-csat-thankyou" style="
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif !important;
                        font-size: 14px !important;
                        font-weight: 500 !important;
                        color: #5f6368 !important;
                        text-align: center !important;
                        padding: 4px !important;
                        margin: 0 !important;
                    ">${thankYouText}</div>
                `;
                
                // Remove after 3 seconds
                setTimeout(() => {
                    if (csatContainer && csatContainer.parentNode) {
                        csatContainer.parentNode.removeChild(csatContainer);
                    }
                }, 3000);
            } else {
                console.error('Error submitting CSAT feedback:', data.message);
            }
        })
        .catch(error => {
            console.error('Error submitting CSAT feedback:', error);
        });
    }

    function resetChatToggle() {
        const chatToggle = document.getElementById('jkf-chat-toggle');
        chatToggle.classList.remove('expanded');
        isToggleExpanded = false;
        
        const cylinder = chatToggle.querySelector('.jkf-cylinder');
        if (cylinder) {
            cylinder.style.width = '60px';
        }
        
        const toggleText = chatToggle.querySelector('.jkf-toggle-text');
        if (toggleText) {
            toggleText.style.opacity = '0';
        }
    }

    function toggleFullScreenMobile() {
        const chatWidget = document.getElementById('jkf-chatbot-widget');
        const chatToggle = document.getElementById('jkf-chat-toggle');
    
        if (chatWidget.style.display === 'none' || chatWidget.style.display === '') {
            // Opening chat
            chatWidget.style.display = 'flex';
            chatWidget.classList.add('open');
            resetChatToggle();
            adjustChatbotSize();
            scrollToBottom();
            
            // Only modify body overflow on mobile devices
            if (window.innerWidth <= 768) {
                document.body.style.overflow = 'hidden';
            }
        } else {
            // Proceed with closing
            chatWidget.style.display = 'none';
            chatWidget.classList.remove('open');
            document.body.style.overflow = '';
            resetChatToggle();
            isActiveConversation = false;
        }
        saveChatState();
    }

    function expandChatToggle() {
        if (!isToggleExpanded && isExpansionScheduled) {
            const chatToggle = document.getElementById('jkf-chat-toggle');
            chatToggle.classList.add('expanded');
            isToggleExpanded = true;
    
            const cylinder = chatToggle.querySelector('.jkf-cylinder');
            if (cylinder) {
                if (chatToggle.classList.contains('left-side')) {
                    cylinder.style.left = '60px'; // Align with the left edge of the circle
                    cylinder.style.right = 'auto';
                } else {
                    cylinder.style.right = '0';
                    cylinder.style.left = 'auto';
                }
            }
        }
    }

    function adjustChatbotSize() {
        const chatWidget = document.getElementById('jkf-chatbot-widget');
        const chatToggle = document.getElementById('jkf-chat-toggle');
        const viewportHeight = window.innerHeight;
        const viewportWidth = window.innerWidth;
        const isDesktop = isDesktopDevice();
        
        if (!isDesktop && viewportWidth <= 768) {
            // Mobile layout
            chatWidget.style.width = '100%';
            chatWidget.style.height = '100%';
            chatWidget.style.bottom = '0';
            chatWidget.style.left = '0';
            chatWidget.style.right = '0';
            chatWidget.style.borderRadius = '0';
            chatToggle.style.bottom = '20px';
            chatToggle.style.right = '20px';
        } else {
            // Desktop layout
            chatWidget.style.width = '420px';
            chatWidget.style.borderRadius = '10px';
            
            if (viewportHeight < 700) {
                chatWidget.style.height = `${viewportHeight - 110}px`;
            } else {
                chatWidget.style.height = '640px';
            }
    
            const position = getComputedStyle(document.documentElement).getPropertyValue('--jkf-chatbot-position').trim();
            
            // Remove all position classes
            chatWidget.classList.remove('left-side', 'right-side');
            chatToggle.classList.remove('left-side', 'right-side');
    
            // Apply positioning based on configuration
            switch (position) {
                case 'bottom-left':
                case 'bottom-left-left':
                    chatWidget.style.left = position === 'bottom-left' ? '20px' : '100px';
                    chatWidget.style.right = 'auto';
                    chatToggle.style.left = position === 'bottom-left' ? '20px' : '100px';
                    chatToggle.style.right = 'auto';
                    chatWidget.classList.add('left-side');
                    chatToggle.classList.add('left-side');
                    break;
                case 'bottom-right-right':
                default: // bottom-right
                    chatWidget.style.right = position === 'bottom-right-right' ? '100px' : '20px';
                    chatWidget.style.left = 'auto';
                    chatToggle.style.right = position === 'bottom-right-right' ? '100px' : '20px';
                    chatToggle.style.left = 'auto';
                    chatWidget.classList.add('right-side');
                    chatToggle.classList.add('right-side');
            }
    
            chatWidget.style.bottom = '100px';
            chatToggle.style.bottom = '20px';
        }
    }
    
    // Old feedback system removed - replaced with inline CSAT

    function triggerChatReset() {
        localStorage.setItem(getClientStorageKey('jkfChatReset'), Date.now().toString());
        resetChat();
    }

    function resetChat() {
        const chatMessages = document.getElementById('jkf-chat-messages');
        const welcomeMessageElement = document.getElementById('jkf-welcome-message');
        const welcomeMessageText = welcomeMessageElement ? welcomeMessageElement.textContent : 'Velkommen! Hvad kan jeg hjælpe med?';

        chatMessages.innerHTML = `
            <div class="jkf-message-container jkf-assistant-message-container">
                <div class="jkf-message jkf-assistant-message" id="jkf-welcome-message">${welcomeMessageText}</div>
            </div>
        `;
        applyWelcomeMessageMargin();

        // Show quick questions again
        renderQuickQuestions();

        const oldThreadId = threadId;

        // Clear state immediately — null out threadId so no messages can be sent
        // against the old thread while we wait for the server to issue a new one.
        clearChatState();
        threadId = null;
        userHasInteracted = false;
        isFeedbackShown = false;
        isActiveConversation = false;

        const baseUrl = window.CHATBOT_BASE_URL || window.location.origin;

        fetch(`${baseUrl}/clear_thread`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ thread_id: oldThreadId }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                threadId = data.thread_id;
                sessionStartTime = new Date().getTime();
                saveChatState();
            }
        })
        .catch(error => console.error('Error resetting chat:', error));

        // Close the chatbot if it's open
        const chatWidget = document.getElementById('jkf-chatbot-widget');
        if (chatWidget.style.display !== 'none') {
            toggleFullScreenMobile();
        }
    }

    function updateWelcomeMessageMargin(config) {
        const welcomeMessage = document.getElementById('jkf-welcome-message');
        if (welcomeMessage && config.welcome_message_margin) {
            document.documentElement.style.setProperty('--jkf-welcome-message-margin', config.welcome_message_margin);
            welcomeMessage.style.marginTop = config.welcome_message_margin;
        }
    }

    function applyWelcomeMessageMargin() {
        const welcomeMessage = document.getElementById('jkf-welcome-message');
        const marginValue = getComputedStyle(document.documentElement).getPropertyValue('--jkf-welcome-message-margin').trim();
        if (welcomeMessage && marginValue) {
            welcomeMessage.style.marginTop = marginValue;
        }
    }

    function initAutoOpenTracking() {
        const sessionKey = `${clientName}_session`;
        const firstVisitKey = `${clientName}_first_visit`;
        const pageCountKey = `${clientName}_page_count`;
        
        // Check if this is the first visit
        if (!localStorage.getItem(firstVisitKey)) {
            localStorage.setItem(firstVisitKey, 'true');
        }
        
        // Initialize or increment page count for the session
        let pageCount = sessionStorage.getItem(pageCountKey);
        if (!pageCount) {
            pageCount = 1;
        } else {
            pageCount = parseInt(pageCount) + 1;
        }
        sessionStorage.setItem(pageCountKey, pageCount);
    }
    
    function shouldAutoOpen(config) {
        if (!config.auto_open) return false;
        
        const firstVisitKey = `${clientName}_first_visit`;
        const pageCountKey = `${clientName}_page_count`;
        
        // Check first visit only setting
        if (config.auto_open_first_visit_only) {
            if (localStorage.getItem(firstVisitKey) !== 'true') {
                return false;
            }
        }
        
        // Check page limit
        if (config.auto_open_page_limit > 0) {
            const pageCount = parseInt(sessionStorage.getItem(pageCountKey) || '1');
            if (pageCount > config.auto_open_page_limit) {
                return false;
            }
        }
        
        return !hasAutoOpened && isDesktopDevice();
    }

    function initChatbot(clientNameParam) {
        clientName = clientNameParam;
        const chatWidget = document.getElementById('jkf-chatbot-widget');
        const chatToggle = document.getElementById('jkf-chat-toggle');
        // Keep both hidden initially
        chatWidget.style.display = 'none';
        chatToggle.style.display = 'none';
        
        const cylinder = document.createElement('div');
        cylinder.className = 'jkf-cylinder';
        const toggleText = document.createElement('span');
        toggleText.className = 'jkf-toggle-text';
        toggleText.textContent = '👋 Brug for hjælp?';
        cylinder.appendChild(toggleText);
        chatToggle.insertBefore(cylinder, chatToggle.firstChild);

        const chatMessages = document.getElementById('jkf-chat-messages');
        chatMessages.addEventListener('scroll', function() {
            userScrollPosition = this.scrollTop;
            shouldAutoScroll = isUserAtBottom();
        });

        const baseUrl = window.CHATBOT_BASE_URL || window.location.origin;

        fetch(`${baseUrl}/chatbot_config`)
            .then(response => response.json())
            .then(config => {
                window.jkfConfig = config;
                // Apply styles
                document.documentElement.style.setProperty('--jkf-primary-color', config.primary_color);
                document.documentElement.style.setProperty('--jkf-toggle-color', config.toggle_color || config.primary_color);
                document.documentElement.style.setProperty('--jkf-chatbot-position', config.chatbot_position || 'bottom-right');
                
                document.querySelector('.jkf-header-logo').src = config.logo_url;

                leadFormButtonText = config.lead_form_button_text || 'Bliv kontaktet';
                leadFormThankYouTitle = config.lead_form_thank_you_title || 'Mange tak';
                leadFormThankYouMessage = config.lead_form_thank_you_message || 'Vi kontakter dig hurtigst muligt';
                newsletterSignup = config.newsletter_signup || false;

                leadFormEnabled = config.lead_form_enabled || false;
                leadFormFields = config.lead_form_fields || [];
                leadFormTitle = config.lead_form_title || 'Dine oplysninger';
                leadFormDescription = config.lead_form_description || '';

                const welcomeMessageElement = document.getElementById('jkf-welcome-message');
                if (welcomeMessageElement) {
                    const storedState = JSON.parse(localStorage.getItem(getClientStorageKey('jkfChatState')));
                    if (storedState && storedState.welcomeMessage) {
                        welcomeMessageElement.textContent = storedState.welcomeMessage;
                    } else {
                        welcomeMessageElement.textContent = config.welcome_message || 'Velkommen! Hvad kan jeg hjælpe med?';
                    }
                }
                
                if (config.welcome_message_margin) {
                    document.documentElement.style.setProperty('--jkf-welcome-message-margin', config.welcome_message_margin);
                    applyWelcomeMessageMargin();
                }
                
                const leadFormButton = document.querySelector('.jkf-lead-form-button');
                if (leadFormEnabled && leadFormFields.length > 0) {
                    leadFormButton.style.display = 'flex';
                    const img = leadFormButton.querySelector('img');
                    if (img) {
                        // Convert hex to RGB for the SVG filter
                        const hexToRgb = (hex) => {
                            const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
                            return result ? {
                                r: parseInt(result[1], 16),
                                g: parseInt(result[2], 16),
                                b: parseInt(result[3], 16)
                            } : null;
                        };
                        
                        const rgb = hexToRgb(config.lead_form_button_color);
                        if (rgb) {
                            img.style.filter = `brightness(0) saturate(100%) invert(${rgb.r/255}) sepia(${rgb.g/255}) saturate(${rgb.b/255})`;
                        }
                    }
                    leadFormButton.style.backgroundColor = config.lead_form_button_background || '#FAFAFA';
                    leadFormButton.addEventListener('click', toggleLeadForm);
                } else {
                    leadFormButton.style.display = 'none';
                }

                const chatbotNameElement = document.getElementById('jkf-chatbot-name');
                if (chatbotNameElement) {
                    chatbotNameElement.textContent = config.chatbot_name || 'AI-assistent';
                }
                
                const disclaimerElement = document.getElementById('jkf-disclaimer');
                if (disclaimerElement) {
                    disclaimerElement.textContent = config.disclaimer_text || 'Dette er kun vejledende svar';
                }

                document.getElementById('jkf-chatbot-widget').classList.add('no-powered-by');

                if (loadChatState() && isActiveConversation) {
                    setTimeout(() => {
                        toggleFullScreenMobile();
                        scrollToBottom();
                    }, 1000);
                }
                
                // Adjust size and position after applying styles
                adjustChatbotSize();

                updateWelcomeMessageMargin(config);
                
                // Load quick questions from config
                if (config.quick_questions && Array.isArray(config.quick_questions)) {
                    quickQuestions = config.quick_questions;
                }
                
                // Show the toggle button after applying styles
                chatToggle.style.display = 'block';
                
                // Render quick questions
                renderQuickQuestions();
                
                isExpansionScheduled = true;
                setTimeout(expandChatToggle, 2000);

                // Apply disclaimer styles
                document.documentElement.style.setProperty('--jkf-disclaimer-color', config.disclaimer_text_color || '#A0ABB6');
                document.documentElement.style.setProperty('--jkf-disclaimer-weight', config.disclaimer_text_bold ? 'bold' : 'normal');
                
                // Apply lead form button styles
                document.documentElement.style.setProperty('--jkf-lead-form-button-color', config.lead_form_button_color || '#A0ABB6');
                document.documentElement.style.setProperty('--jkf-lead-form-button-background', config.lead_form_button_background || '#FAFAFA');
                
                initAutoOpenTracking();
                
                if (shouldAutoOpen(config)) {
                    setTimeout(() => {
                        toggleFullScreenMobile();
                        hasAutoOpened = true;
                        if (config.auto_open_first_visit_only) {
                            localStorage.setItem(`${clientName}_first_visit`, 'false');
                        }
                    }, 1000);
                }
            })
            .catch(error => console.error('Error fetching client config:', error));

        initChat();
    }

    function addViewportMeta() {
        const viewportMeta = document.createElement('meta');
        viewportMeta.name = 'viewport';
        viewportMeta.content = 'width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no';
        document.head.appendChild(viewportMeta);
    }

    // Function to check session expiration
    function checkSessionExpiration() {
        if (sessionStartTime) {
            const currentTime = new Date().getTime();
            if (currentTime - sessionStartTime >= SESSION_DURATION) {
                clearChatState();
                initChat();
                resetChat();
            }
        }
    }

    // Event Listeners
    function addEventListeners() {
        window.addEventListener('resize', adjustChatbotSize);

        document.getElementById('jkf-chat-toggle').addEventListener('click', function(event) {
            const chatToggle = this;
            const cylinder = chatToggle.querySelector('.jkf-cylinder');
            
            if (cylinder) {
                const cylinderRect = cylinder.getBoundingClientRect();
                const isClickOnCylinder = 
                    event.clientX >= cylinderRect.left && 
                    event.clientX <= cylinderRect.right && 
                    event.clientY >= cylinderRect.top && 
                    event.clientY <= cylinderRect.bottom;

                if (isToggleExpanded && !isClickOnCylinder) {
                    chatToggle.classList.remove('expanded');
                    isToggleExpanded = false;
                } else {
                    if (!isToggleExpanded) {
                        isExpansionScheduled = false;
                    }
                    toggleFullScreenMobile();
                }
            } else {
                toggleFullScreenMobile();
            }
        });

        document.getElementById('jkf-close-chat').addEventListener('click', function() {
            const chatWidget = document.getElementById('jkf-chatbot-widget');
            chatWidget.style.display = 'none';
            chatWidget.classList.remove('open');
            document.body.style.overflow = '';
            resetChatToggle();
        });

        document.getElementById('jkf-restart-chat').addEventListener('click', function() {
            const chatMessages = document.getElementById('jkf-chat-messages');
            const welcomeMessageElement = document.getElementById('jkf-welcome-message');
            const welcomeMessageText = welcomeMessageElement ? welcomeMessageElement.textContent : 'Velkommen! Hvad kan jeg hjælpe med?';
        
            chatMessages.innerHTML = `
                <div class="jkf-message-container jkf-assistant-message-container">
                    <div class="jkf-message jkf-assistant-message" id="jkf-welcome-message">${welcomeMessageText}</div>
                </div>
            `;
            applyWelcomeMessageMargin();
            
            // Show quick questions again
            renderQuickQuestions();
            
            // Clear CSAT state when restarting chat
            const oldThreadId = threadId;
            clearChatState();
            threadId = null;
            userHasInteracted = false;
            isFeedbackShown = false;

            const baseUrl = window.CHATBOT_BASE_URL || window.location.origin;

            fetch(`${baseUrl}/clear_thread`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ thread_id: oldThreadId }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    threadId = data.thread_id;
                    sessionStartTime = new Date().getTime();
                    saveChatState();
                }
            })
            .catch(error => console.error('Error restarting chat:', error));
        });

        document.getElementById('jkf-user-input').addEventListener('input', function() {
            const sendArrow = document.querySelector('.jkf-send-arrow');
            sendArrow.style.visibility = this.value.trim() !== '' ? 'visible' : 'hidden';
            
            // Reset inactivity timer when user types
            resetInactivityTimer();
        });

        document.getElementById('jkf-user-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !isAssistantResponding) {
                sendMessage();
            }
        });

        document.querySelector('.jkf-send-arrow').addEventListener('click', function() {
            if (!isAssistantResponding) {
                sendMessage();
            }
        });

        const leadFormButton = document.querySelector('.jkf-lead-form-button');
        if (leadFormButton) {
            leadFormButton.addEventListener('click', toggleLeadForm);
        }
    }

    // Main initialization function
    async function initialize() {
        // For JKF (single-tenant), CHATBOT_CLIENT_NAME defaults to 'jkf'
        if (!window.CHATBOT_CLIENT_NAME) {
            window.CHATBOT_CLIENT_NAME = 'jkf';
        }

        // Read HMAC-signed company token from JKF Universe (null for anonymous users)
        companyId = window.CHATBOT_COMPANY_TOKEN || null;

        try {
            addViewportMeta();
            await loadCSS(CSS_URL);
            createChatbotHTML();
            initChatbot(window.CHATBOT_CLIENT_NAME);
            addEventListeners();
            addUrlClickListeners();
            
            setInterval(checkSessionExpiration, 60000);
    
            window.addEventListener('storage', function(e) {
                if (e.key === getClientStorageKey('jkfChatReset')) {
                    resetChat();
                }
            });
        } catch (error) {
            console.error('Error initializing chatbot:', error);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize);
    } else {
        initialize();
    }
})();
