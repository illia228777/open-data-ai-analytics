provider "aws" {
  region = var.region
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
}

resource "aws_key_pair" "deployer" {
  key_name   = var.key_name
  public_key = file("~/.ssh/lab-key.pub")
}

resource "aws_security_group" "main" {
  name        = "analytics-sg"
  description = "Allow SSH and web access"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Web app"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

   ingress {
       description = "Grafana"
       from_port   = 3000
       to_port     = 3000
       protocol    = "tcp"
       cidr_blocks = ["0.0.0.0/0"]
   }

   ingress {
       description = "Prometheus"
       from_port   = 9090
       to_port     = 9090
       protocol    = "tcp"
       cidr_blocks = ["0.0.0.0/0"]
   }

ingress {
    description = "K8s Web App"
    from_port   = 30080
    to_port     = 30080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Argo CD"
    from_port   = 30443
    to_port     = 30443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "main" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.deployer.key_name
  vpc_security_group_ids = [aws_security_group.main.id]
  user_data              = file("cloud-init.yaml")

  root_block_device {
    volume_size = 20
  }

  tags = { Name = "analytics-lab" }
}
