# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
RUN apt update && \
    apt install -y libgl1-mesa-glx
# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World


EXPOSE $PORT
#CMD ["gunicorn", "main:app", "-b", "0.0.0.0:$PORT"]
CMD ["python", "main.py"]



# Run app.py when the container launches
#CMD ["gunicorn", "main:app", "-b", "0.0.0.0:5000"]
