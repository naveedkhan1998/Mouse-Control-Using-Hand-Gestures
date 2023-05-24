# Mouse-Control-Using-Hand-Gestures

This project aims to control the mouse cursor on a computer using a webcam. By tracking the movement of the user's hand or a specific object in front of the webcam, the program will interpret the motion and translate it into corresponding cursor movements.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [License](#license)

## Prerequisites

To run this project, you will need the following:

- A computer with a webcam
- Python installed (version 3.8.8)
- Required Python libraries (virtualenv)

## Installation

Follow the steps below to set up the project:

1. Clone the repository:

   ```bash
   git clone https://github.com/naveedkhan1998/Mouse-Control-Using-Hand-Gestures.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Mouse-Control-Using-Hand-Gestures
   ```

3. Create a virtual environment and activate it:

   ```bash
   pip install virtualenv
   virtualenv venv --python=python3.8.8
   . .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

1. Connect your webcam to the computer.

2. Run the main Python script while the virtual enivronment is running:

   ```bash
   python main.py
   ```

3. The webcam feed will appear in a new window.

4. Place your hand in front of the webcam.

5. Move your hand or object to control the mouse cursor on the screen.

6. Press the `q` key to exit the program.

## Contributing

Contributions to this project are welcome. To contribute, follow these steps:

1. Fork the repository.

2. Create a new branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your modifications and commit them:

   ```bash
   git commit -m "Add your commit message"
   ```

4. Push to the branch:

   ```bash
   git push origin feature/your-feature-name
   ```

5. Create a pull request with a detailed description of your changes.

## License

This project is licensed under the [MIT License](LICENSE). You are free to modify and distribute the code. Please see the [LICENSE](LICENSE) file for more details.
