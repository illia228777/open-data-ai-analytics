output "public_ip" {
  value = aws_instance.main.public_ip
}

output "web_url" {
  value = "http://${aws_instance.main.public_ip}:8000"
}
