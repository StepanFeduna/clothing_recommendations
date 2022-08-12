FROM python:3.8.13-stretch

# Maintainer info
LABEL maintainer="feduna.stepan@gmail.com"

# Make working directories
RUN  mkdir -p  /clothing_recommendations
WORKDIR  /clothing_recommendations

# Upgrade pip with no cache
RUN pip install --no-cache-dir -U pip

# Copy application requirements file to the created working directory
COPY requirements.txt .

# Install application dependencies from the requirements file
RUN pip install -r requirements.txt

# Copy every file in the source folder to the created working directory
COPY  . .

# Run the python application
CMD ["python", "main.py"]