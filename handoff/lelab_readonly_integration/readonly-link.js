(() => {
  if (document.getElementById('so101-readonly-link')) return;
  const a = document.createElement('a');
  a.id = 'so101-readonly-link'; a.href = '/so101-readonly';
  a.textContent = '10k 只读预测';
  a.style.cssText = 'position:fixed;right:22px;bottom:76px;z-index:10000;padding:12px 18px;border-radius:12px;background:#087f8c;color:white;font:600 14px system-ui;text-decoration:none;box-shadow:0 3px 14px #0003';
  document.body.appendChild(a);
})();
