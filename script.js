document.addEventListener('DOMContentLoaded', () => {
    const openBtn = document.getElementById('openModal');
    const closeBtn = document.getElementById('closeModal');
    const backdrop = document.getElementById('modalBackdrop');
    const actionBtn = document.getElementById('modalAction');

    const toggleModal = () => {
        backdrop.classList.toggle('active');
        if (backdrop.classList.contains('active')) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }
    };

    openBtn.addEventListener('click', toggleModal);
    closeBtn.addEventListener('click', toggleModal);

    // Close on backdrop click
    backdrop.addEventListener('click', (e) => {
        if (e.target === backdrop) {
            toggleModal();
        }
    });

    // Action button feedback
    actionBtn.addEventListener('click', () => {
        actionBtn.textContent = 'Refining...';
        setTimeout(() => {
            alert('Experience Optimized!');
            actionBtn.textContent = 'Learn More';
            toggleModal();
        }, 1000);
    });

    // Escape key handling
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && backdrop.classList.contains('active')) {
            toggleModal();
        }
    });
});
