package demo

interface MathOperations {
    fun add(a: Int, b: Int): Int
}

enum class Operation {
    ADD,
    SUBTRACT,
}

data class Calculator(val name: String) : MathOperations {
    companion object {
        fun create(): Calculator = Calculator("generated")
    }

    constructor() : this("default")

    override fun add(a: Int, b: Int): Int {
        return a + b
    }

    val version: String = "1.0"
}
