/**
 * Heart Disease Clinical AI Suite - Interactive Helper Scripts
 */
document.addEventListener("DOMContentLoaded", () => {
    console.log("🫀 Heart Disease Clinical AI Suite Initialized");
    
    // Add smooth hover transitions to all cards
    const cards = document.querySelectorAll(".clinical-card");
    cards.forEach(card => {
        card.addEventListener("mouseenter", () => {
            card.style.borderColor = "rgba(56, 189, 248, 0.4)";
        });
        card.addEventListener("mouseleave", () => {
            card.style.borderColor = "rgba(255, 255, 255, 0.08)";
        });
    });
});
