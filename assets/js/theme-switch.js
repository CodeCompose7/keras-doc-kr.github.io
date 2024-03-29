window.addEventListener("DOMContentLoaded", function () {
    // 새로운 a 요소를 생성합니다.
   var newLink = document.createElement('a');
   newLink.className = 'site-button';
   newLink.id = 'theme-toggle';

   // SVG 요소를 생성합니다.
   var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
   svg.setAttribute('width', '18px');
   svg.setAttribute('height', '18px');

   // use 요소를 생성합니다.
   var use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
   use.setAttribute('href', '#svg-sun');

   // SVG 요소를 조립합니다.
   svg.appendChild(use);

   // SVG 요소를 a 요소에 추가합니다.
   newLink.appendChild(svg);

   // ul 요소를 찾습니다.
   var ulList = document.querySelector('.aux-nav-list');

   // ul 요소의 첫 번째 자식으로 새로운 a 요소를 추가합니다.
   ulList.prepend(newLink);
   
    const toggleDarkMode = document.getElementById("theme-toggle");

    if (localStorage.getItem('theme') === 'dark') {
        setTheme('dark');
    } else {
        setTheme('light');
    }

    jtd.addEvent(toggleDarkMode, 'click', function () {
        const currentTheme = getTheme();
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        localStorage.setItem('theme', newTheme);
        setTheme(newTheme);
    });

    function getTheme() {
        return document.documentElement.classList.contains('dark-mode') ? 'dark' : 'light';
    }

    function setTheme(theme) {
        if (theme === 'dark') {
            toggleDarkMode.innerHTML = `<svg width='18px' height='18px'><use href="#svg-moon"></use></svg>`;
            document.documentElement.classList.add('dark-mode');
            document.documentElement.classList.remove('light-mode');
        } else {
            toggleDarkMode.innerHTML = `<svg width='18px' height='18px'><use href="#svg-sun"></use></svg>`;
            document.documentElement.classList.add('light-mode');
            document.documentElement.classList.remove('dark-mode');
        }
    }
});
