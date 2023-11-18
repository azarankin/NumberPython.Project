# Use a base image with the desired GLIBC version
FROM debian:bullseye-slim

# Install necessary system libraries
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port on which your application will run
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "main:app", "-b", "0.0.0.0:5000"]