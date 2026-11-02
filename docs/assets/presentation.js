(() => {
  'use strict';
  const themeButton = document.querySelector('#theme-toggle');
  const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
  let chosenTheme = null;
  try {
    const saved = localStorage.getItem('erig-color-scheme');
    if (['light', 'dark'].includes(saved)) chosenTheme = saved;
  } catch { /* Storage may be disabled; the theme still works for this visit. */ }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]').content = theme === 'dark' ? '#27272a' : '#ffffff';
    if (themeButton) {
      const label = theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme';
      themeButton.setAttribute('aria-label', label);
      themeButton.title = label;
    }
  }

  applyTheme(chosenTheme || (systemTheme.matches ? 'dark' : 'light'));
  systemTheme.addEventListener('change', () => {
    if (!chosenTheme) applyTheme(systemTheme.matches ? 'dark' : 'light');
  });
  if (themeButton) {
    themeButton.hidden = false;
    themeButton.addEventListener('click', () => {
      chosenTheme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      applyTheme(chosenTheme);
      try { localStorage.setItem('erig-color-scheme', chosenTheme); } catch { /* Optional persistence. */ }
    });
  }

  const chapters = [...document.querySelectorAll('main > .chapter')];
  const presentButton = document.querySelector('#present');
  const controls = document.querySelector('#presentation-controls');
  const position = document.querySelector('#slide-position');
  const previous = document.querySelector('#previous-slide');
  const next = document.querySelector('#next-slide');
  const slideNumber = document.querySelector('#slide-number');
  const sectionLinks = [...document.querySelectorAll('[data-section-link]')];
  let presenting = false;
  let slide = 0;

  function updateSectionNavigation(index) {
    const current = chapters[index];
    sectionLinks.forEach(link => {
      const group = chapters.filter(chapter => chapter.dataset.section === link.dataset.sectionLink);
      const active = current.dataset.section === link.dataset.sectionLink;
      if (active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
      link.querySelector('.section-progress').textContent = active
        ? `${group.indexOf(current) + 1} / ${group.length}`
        : `${group.length} ${group.length === 1 ? 'slide' : 'slides'}`;
    });
  }

  function indexFromHash() {
    return chapters.findIndex(chapter => `#${chapter.id}` === location.hash);
  }

  function showSlide(index, updateHash = true) {
    slide = Math.max(0, Math.min(chapters.length - 1, index));
    chapters.forEach((chapter, i) => chapter.classList.toggle('is-active', i === slide));
    position.textContent = chapters[slide].dataset.title;
    slideNumber.value = slide + 1;
    updateSectionNavigation(slide);
    previous.disabled = slide === 0;
    next.disabled = slide === chapters.length - 1;
    // replaceState keeps slide navigation from filling browser history.
    if (updateHash) history.replaceState(null, '', `#${chapters[slide].id}`);
    const heading = chapters[slide].querySelector('h1, h2');
    heading.tabIndex = -1;
    heading.focus({preventScroll: true});
    window.scrollTo({top: 0, behavior: 'instant'});
  }

  function setPresentation(enabled) {
    if (enabled) {
      const nearest = chapters.reduce((best, chapter, i) =>
        Math.abs(chapter.getBoundingClientRect().top - document.querySelector('.site-header').offsetHeight) < Math.abs(chapters[best].getBoundingClientRect().top - document.querySelector('.site-header').offsetHeight) ? i : best, 0);
      slide = nearest;
    }
    presenting = enabled;
    document.body.classList.toggle('presenting', enabled);
    controls.hidden = !enabled;
    presentButton.setAttribute('aria-pressed', String(enabled));
    presentButton.textContent = enabled ? 'Exit presentation' : 'Present ↗';
    if (enabled) showSlide(slide);
    else {
      chapters.forEach(chapter => chapter.classList.remove('is-active'));
      chapters[slide].scrollIntoView({behavior: 'instant'});
      presentButton.focus({preventScroll: true});
    }
  }

  if (presentButton && chapters.length) {
    slideNumber.max = chapters.length;
    document.querySelector('#slide-total').textContent = `/ ${chapters.length}`;
    function jumpToSlide() {
      const number = slideNumber.valueAsNumber;
      if (Number.isInteger(number) && number >= 1 && number <= chapters.length) showSlide(number - 1);
      else slideNumber.value = slide + 1;
    }
    slideNumber.addEventListener('change', jumpToSlide);
    slideNumber.addEventListener('keydown', event => {
      if (event.key === 'Enter') { event.preventDefault(); jumpToSlide(); }
    });
    presentButton.hidden = false;
    presentButton.addEventListener('click', () => setPresentation(!presenting));
    previous.addEventListener('click', () => showSlide(slide - 1));
    next.addEventListener('click', () => showSlide(slide + 1));
    document.querySelector('#exit-presentation').addEventListener('click', () => setPresentation(false));
    document.addEventListener('keydown', event => {
      if (!presenting) return;
      if (event.key === 'Escape') { setPresentation(false); return; }
      if (event.target.closest('input, textarea, select, button, a, [contenteditable]')) return;
      if (['ArrowRight', 'PageDown', ' '].includes(event.key)) { event.preventDefault(); showSlide(slide + 1); }
      if (['ArrowLeft', 'PageUp'].includes(event.key)) { event.preventDefault(); showSlide(slide - 1); }
      if (event.key === 'Home') { event.preventDefault(); showSlide(0); }
      if (event.key === 'End') { event.preventDefault(); showSlide(chapters.length - 1); }
    });
    // Handle both changed hashes and links to the already-current hash.
    document.addEventListener('click', event => {
      const link = event.target.closest('a[href^="#"]');
      if (!presenting || !link) return;
      const index = chapters.findIndex(chapter => `#${chapter.id}` === link.getAttribute('href'));
      if (index >= 0) { event.preventDefault(); showSlide(index); }
    });
    window.addEventListener('hashchange', () => {
      const index = indexFromHash();
      if (presenting && index >= 0) showSlide(index, false);
    });
    // Reading mode tracks the section at the top of the content area as well.
    let scrollQueued = false;
    window.addEventListener('scroll', () => {
      if (presenting || scrollQueued) return;
      scrollQueued = true;
      requestAnimationFrame(() => {
        scrollQueued = false;
        if (presenting) return;
        const top = document.querySelector('.site-header').offsetHeight + 40;
        let current = 0;
        chapters.forEach((chapter, i) => { if (chapter.getBoundingClientRect().top <= top) current = i; });
        updateSectionNavigation(current);
      });
    }, {passive: true});
    updateSectionNavigation(Math.max(0, indexFromHash()));
  }

  const slider = document.querySelector('#temperature-control');
  if (slider) {
    const gradients = [0.15, 0.4, 0.9, 1.8, 2.2, 1.2, 0.6, 0.25];
    const ns = 'http://www.w3.org/2000/svg';
    const group = document.querySelector('#weight-bars');
    const bars = gradients.map((gradient, i) => {
      const bar = document.createElementNS(ns, 'rect');
      bar.setAttribute('x', 26 + i * 50);
      bar.setAttribute('width', '30');
      bar.setAttribute('fill', i === 4 ? 'var(--demo-highlight)' : 'var(--demo-bar)');
      const label = document.createElementNS(ns, 'text');
      label.setAttribute('x', 41 + i * 50);
      label.setAttribute('y', '232');
      label.setAttribute('text-anchor', 'middle');
      label.setAttribute('class', 'chart-axis');
      label.textContent = `g${i + 1}`;
      group.append(bar, label);
      return bar;
    });
    function updateWeights(temperature) {
      const scores = gradients.map(g => g * g / temperature);
      const max = Math.max(...scores);
      const exponentials = scores.map(score => Math.exp(score - max));
      const total = exponentials.reduce((sum, value) => sum + value, 0);
      const weights = exponentials.map(value => value / total);
      weights.forEach((weight, i) => {
        const height = weight * 175;
        bars[i].setAttribute('height', height);
        bars[i].setAttribute('y', 212 - height);
      });
      document.querySelector('#temperature-value').value = `τ = ${temperature.toFixed(2)}`;
      slider.setAttribute('aria-valuetext', `Temperature ${temperature.toFixed(2)}`);
      const text = `Largest feature weight: ${(Math.max(...weights) * 100).toFixed(1)}%. Uniform: 12.5% each.`;
      document.querySelector('#weight-summary').textContent = text;
      document.querySelector('#weights-desc').textContent = `Temperature ${temperature.toFixed(2)}. Feature weights: ${weights.map(w => `${(w * 100).toFixed(1)}%`).join(', ')}.`;
    }
    slider.disabled = false;
    slider.addEventListener('input', () => updateWeights(10 ** Number(slider.value)));
    const reset = document.querySelector('#reset-temperature');
    reset.disabled = false;
    reset.addEventListener('click', () => { slider.value = Math.log10(3); updateWeights(3); });
    updateWeights(3);
  }

  document.querySelectorAll('[data-copy]').forEach(button => {
    button.hidden = false;
    button.addEventListener('click', async () => {
      const code = document.getElementById(button.dataset.copy);
      try {
        if (!navigator.clipboard) throw new Error('Clipboard unavailable');
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = 'Copied';
        document.querySelector('#copy-status').textContent = 'Code copied to clipboard.';
      } catch {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(code);
        selection.removeAllRanges();
        selection.addRange(range);
        button.textContent = 'Selected';
        document.querySelector('#copy-status').textContent = 'Code selected. Use your browser’s copy command.';
      }
      window.setTimeout(() => { button.textContent = 'Copy'; }, 2200);
    });
  });
})();
