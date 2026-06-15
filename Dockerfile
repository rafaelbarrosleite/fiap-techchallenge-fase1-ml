FROM python:3.11-slim

# Evita perguntas interativas e deixa o log do Python sair na hora
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instala as dependências primeiro (aproveita o cache do Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do projeto (notebook, dataset, etc.)
COPY . .

# Porta padrão do Jupyter
EXPOSE 8888

# Sobe o Jupyter acessível de fora do container
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
