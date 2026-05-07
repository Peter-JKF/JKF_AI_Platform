/**
 * ConvoChat Dashboard - Main Scripts
 * Modern Dashboard UI Interactions
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initSidebar();
    initNavSubsections();
    initDropdowns();
    initActiveNav();
    initDarkMode();
});

/**
 * Sidebar Toggle for Mobile
 */
function initSidebar() {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebarOverlay = document.getElementById('sidebarOverlay');
    const sidebarCollapseBtn = document.getElementById('sidebarCollapseBtn');
    const appContainer = document.querySelector('.app-container');
    
    // Mobile toggle
    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', function() {
            sidebar.classList.toggle('open');
            if (sidebarOverlay) {
                sidebarOverlay.classList.toggle('active');
            }
        });
    }
    
    if (sidebarOverlay && sidebar) {
        sidebarOverlay.addEventListener('click', function() {
            sidebar.classList.remove('open');
            sidebarOverlay.classList.remove('active');
        });
    }
    
    // Desktop collapse toggle
    if (sidebarCollapseBtn && sidebar) {
        // Check if already collapsed (set by inline script in head)
        const isCollapsed = document.documentElement.classList.contains('sidebar-collapsed');
        
        // Set initial button title
        sidebarCollapseBtn.title = isCollapsed ? 'Vis sidebar' : 'Skjul sidebar';
        
        sidebarCollapseBtn.addEventListener('click', function() {
            // Toggle the html class for instant CSS application
            document.documentElement.classList.toggle('sidebar-collapsed');
            
            // Save preference to localStorage
            const nowCollapsed = document.documentElement.classList.contains('sidebar-collapsed');
            localStorage.setItem('sidebarCollapsed', nowCollapsed);
            
            // Update button title
            this.title = nowCollapsed ? 'Vis sidebar' : 'Skjul sidebar';
        });
    }
    
    // Close sidebar when clicking nav items on mobile
    const navItems = document.querySelectorAll('.sidebar .nav-item');
    navItems.forEach(function(item) {
        item.addEventListener('click', function() {
            if (window.innerWidth <= 1024) {
                sidebar.classList.remove('open');
                if (sidebarOverlay) {
                    sidebarOverlay.classList.remove('active');
                }
            }
        });
    });
}

/**
 * Collapsible nav subsections for dual-product sidebar
 * Initial collapsed state is set server-side to avoid flash on reload.
 * Clicking the section link navigates to the first page; the chevron toggle expands/collapses.
 */
function initNavSubsections() {
    var subsections = document.querySelectorAll('.nav-subsection');
    if (!subsections.length) return;

    subsections.forEach(function(section) {
        var key = section.getAttribute('data-subsection');
        var toggleBtn = section.querySelector('.nav-subsection-toggle');
        var hasActive = section.querySelector('.nav-item.active') !== null;

        if (hasActive) {
            section.classList.add('has-active');
        }

        if (toggleBtn) {
            toggleBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                section.classList.toggle('collapsed');
                var isCollapsed = section.classList.contains('collapsed');
                localStorage.setItem('navSubsection_' + key, isCollapsed ? 'collapsed' : 'expanded');
            });
        }
    });
}

/**
 * Initialize Bootstrap Dropdowns
 */
function initDropdowns() {
    // User menu dropdown
    const userMenu = document.getElementById('userMenuDropdown');
    if (userMenu) {
        // Initialize Bootstrap dropdown
        new bootstrap.Dropdown(userMenu);
    }
}

/**
 * Set Active Navigation Item
 */
function initActiveNav() {
    const currentPath = window.location.pathname;
    const navItems = document.querySelectorAll('.sidebar .nav-item');
    
    navItems.forEach(function(item) {
        const href = item.getAttribute('href');
        if (href && currentPath.startsWith(href) && href !== '#') {
            // Remove active from all
            navItems.forEach(function(nav) {
                nav.classList.remove('active');
            });
            // Add active to current
            item.classList.add('active');
        }
    });
}

/**
 * Utility: Format Numbers with Thousands Separator
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".");
}

/**
 * Utility: Truncate Text
 */
function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substr(0, maxLength - 3) + '...';
}

/**
 * Utility: Debounce Function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Utility: Smooth Scroll to Element
 */
function scrollToElement(element, offset = 0) {
    const elementPosition = element.getBoundingClientRect().top;
    const offsetPosition = elementPosition + window.pageYOffset - offset;

    window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth'
    });
}

/**
 * Table Sort Functionality
 */
function initTableSort(tableId) {
    const table = document.getElementById(tableId);
    if (!table) return;
    
    const headers = table.querySelectorAll('th[data-sort]');
    headers.forEach(function(header) {
        header.style.cursor = 'pointer';
        header.addEventListener('click', function() {
            const column = this.dataset.sort;
            const direction = this.dataset.direction === 'asc' ? 'desc' : 'asc';
            
            // Update direction
            headers.forEach(h => h.dataset.direction = '');
            this.dataset.direction = direction;
            
            // Sort table
            sortTable(table, column, direction);
        });
    });
}

function sortTable(table, column, direction) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    rows.sort(function(a, b) {
        const aValue = a.querySelector(`td:nth-child(${parseInt(column) + 1})`).textContent.trim();
        const bValue = b.querySelector(`td:nth-child(${parseInt(column) + 1})`).textContent.trim();
        
        // Try numeric sort first
        const aNum = parseFloat(aValue.replace(/[^\d.-]/g, ''));
        const bNum = parseFloat(bValue.replace(/[^\d.-]/g, ''));
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return direction === 'asc' ? aNum - bNum : bNum - aNum;
        }
        
        // Fall back to string sort
        return direction === 'asc' 
            ? aValue.localeCompare(bValue, 'da')
            : bValue.localeCompare(aValue, 'da');
    });
    
    // Re-append sorted rows
    rows.forEach(row => tbody.appendChild(row));
}

/**
 * Copy to Clipboard
 */
function copyToClipboard(text, successMessage) {
    navigator.clipboard.writeText(text).then(function() {
        if (typeof createToast === 'function') {
            createToast(successMessage || 'Kopieret til udklipsholder', 'success');
        }
    }).catch(function(err) {
        console.error('Could not copy text: ', err);
        if (typeof createToast === 'function') {
            createToast('Kunne ikke kopiere tekst', 'danger');
        }
    });
}

/**
 * Handle Window Resize
 */
window.addEventListener('resize', debounce(function() {
    // Close mobile sidebar on resize to desktop
    if (window.innerWidth > 1024) {
        const sidebar = document.getElementById('sidebar');
        const sidebarOverlay = document.getElementById('sidebarOverlay');
        
        if (sidebar) sidebar.classList.remove('open');
        if (sidebarOverlay) sidebarOverlay.classList.remove('active');
    }
}, 250));

/**
 * Dark Mode Initialization and Control
 */
function initDarkMode() {
    const darkModeToggle = document.getElementById('darkModeToggle');
    const darkModeIcon = document.getElementById('darkModeIcon');
    
    if (!darkModeToggle || !darkModeIcon) return;
    
    // Get current theme from localStorage or check system preference
    const currentTheme = getCurrentTheme();
    
    // Update icon based on current theme
    updateDarkModeIcon(currentTheme);
    
    // Add click handler
    darkModeToggle.addEventListener('click', toggleDarkMode);
}

function getCurrentTheme() {
    const storedTheme = localStorage.getItem('theme');
    if (storedTheme) {
        return storedTheme;
    }
    
    // Check if theme was set in inline script (system preference)
    const htmlTheme = document.documentElement.getAttribute('data-theme');
    if (htmlTheme) {
        return htmlTheme;
    }
    
    return 'light';
}

function toggleDarkMode() {
    const htmlElement = document.documentElement;
    const currentTheme = htmlElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    // Update HTML attribute
    if (newTheme === 'dark') {
        htmlElement.setAttribute('data-theme', 'dark');
    } else {
        htmlElement.removeAttribute('data-theme');
    }
    
    // Save to localStorage
    localStorage.setItem('theme', newTheme);
    
    // Update icon
    updateDarkModeIcon(newTheme);
    
    // Optional: Show a subtle toast notification
    if (typeof createToast === 'function') {
        const message = newTheme === 'dark' ? 'Mørkt tema aktiveret' : 'Lyst tema aktiveret';
        createToast(message, 'success');
    }
}

function updateDarkModeIcon(theme) {
    const darkModeIcon = document.getElementById('darkModeIcon');
    if (!darkModeIcon) return;
    
    // Update icon: moon for light mode (click to go dark), sun for dark mode (click to go light)
    if (theme === 'dark') {
        darkModeIcon.className = 'fas fa-sun';
        const button = document.getElementById('darkModeToggle');
        if (button) button.title = 'Skift til lyst tema';
    } else {
        darkModeIcon.className = 'fas fa-moon';
        const button = document.getElementById('darkModeToggle');
        if (button) button.title = 'Skift til mørkt tema';
    }
}

/**
 * Listen for system theme changes (optional enhancement)
 */
if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
        // Only update if user hasn't manually set a preference
        if (!localStorage.getItem('theme')) {
            const newTheme = e.matches ? 'dark' : 'light';
            if (newTheme === 'dark') {
                document.documentElement.setAttribute('data-theme', 'dark');
            } else {
                document.documentElement.removeAttribute('data-theme');
            }
            updateDarkModeIcon(newTheme);
        }
    });
}
