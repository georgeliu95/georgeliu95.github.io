document.querySelectorAll('[role="tablist"]').forEach(list => {
  const tabs = [...list.querySelectorAll('[role="tab"]')];
  function select(tab) {
    tabs.forEach(item => {
      const active = item === tab;
      item.setAttribute('aria-selected', String(active));
      item.classList.toggle('active', active);
      item.tabIndex = active ? 0 : -1;
      document.getElementById(item.getAttribute('aria-controls')).hidden = !active;
    });
  }
  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => select(tab));
    tab.addEventListener('keydown', event => {
      const offset = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
      if (offset) {
        event.preventDefault();
        const next = tabs[(i + offset + tabs.length) % tabs.length];
        select(next);
        next.focus();
      }
    });
  });
});
