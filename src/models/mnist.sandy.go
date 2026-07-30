func main(x Node) Node {
    x = __linear(x, @fc1.weight, @fc1.bias)
    x = __relu(x)
    x = __linear(x, @fc2.weight, @fc2.bias)
    return x
}
