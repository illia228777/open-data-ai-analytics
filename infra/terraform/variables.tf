variable "region" {
  default = "eu-central-1"
}

variable "instance_type" {
  default = "t3.small"
}

variable "key_name" {
  description = "SSH key pair name"
  default     = "lab-key"
}
