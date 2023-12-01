document.addEventListener("DOMContentLoaded", function () {
    const listDisplay = document.getElementById("listDisplay");
    const insertText = document.getElementById("insertText");
    const insertButton = document.getElementById("insertButton");

    const list = [];

    insertButton.addEventListener("click", function () {
        const value = insertText.value.trim();

        if (value !== "") {
            list.push(value);
            updateListDisplay();
        }

        insertText.value = "";
    });

    function updateListDisplay() {
        listDisplay.innerHTML = "";
        list.forEach(function (item, index) {
            const listItem = document.createElement("div");
            listItem.classList.add("list-item");
    
            if ((index + 1) % 3 === 0) {
                listItem.classList.add("red");
            } else {
                listItem.classList.add("black");
            }
    
            listItem.textContent = item;
            listDisplay.appendChild(listItem);
        });
    }
    
    
});
