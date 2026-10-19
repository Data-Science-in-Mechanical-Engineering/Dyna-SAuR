// Page interactions: video chapters, ablation tabs, BibTeX copy, active nav link.

const videos = Array.from(document.querySelectorAll('video'));

// Only one video plays at a time.
videos.forEach((v) => {
  v.addEventListener('play', () => videos.forEach((o) => { if (o !== v) o.pause(); }));
});

// Chapter cards scroll to their clip and start it.
document.querySelectorAll('[data-play]').forEach((link) => {
  link.addEventListener('click', (e) => {
    const target = document.getElementById(link.dataset.play);
    if (!target) return;
    e.preventDefault();
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    const video = target.querySelector('video');
    if (video) video.play().catch(() => {});
    history.replaceState(null, '', `#${link.dataset.play}`);
  });
});

// Ablation tabs.
document.querySelectorAll('.tabs').forEach((tabs) => {
  const buttons = tabs.querySelectorAll('.tab');
  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      buttons.forEach((b) => {
        const on = b === btn;
        b.classList.toggle('active', on);
        b.setAttribute('aria-selected', String(on));
        document.getElementById(b.dataset.tab).classList.toggle('active', on);
      });
    });
  });
});

// Copy BibTeX.
document.querySelectorAll('[data-copy]').forEach((btn) => {
  btn.addEventListener('click', async () => {
    const text = document.getElementById(btn.dataset.copy).textContent;
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = 'Copied!';
    } catch {
      btn.textContent = 'Press Ctrl+C';
      const range = document.createRange();
      range.selectNodeContents(document.getElementById(btn.dataset.copy));
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }
    setTimeout(() => { btn.textContent = 'Copy'; }, 1800);
  });
});

// Highlight the nav link of the section in view.
const navLinks = Array.from(document.querySelectorAll('.topnav a'));
const sections = navLinks.map((a) => document.querySelector(a.getAttribute('href'))).filter(Boolean);
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    navLinks.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === `#${entry.target.id}`));
  });
}, { rootMargin: '-45% 0px -50% 0px' });
sections.forEach((s) => observer.observe(s));
