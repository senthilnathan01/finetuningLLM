
original_rl_model_prompt = "Find the positive integer n such that C(n,4) = 126."


rl_model_ans = """To find the value of n, we need to solve the equation $\binom{n}{4} = 126$.

The binomial coefficient $\binom{n}{4}$ is defined as $\frac{n!}{4!(n-4)!}$, where $n!$ denotes the factorial of n, which is the product of all positive integers up to n.

So we have $\frac{n!}{4!(n-4)!} = 126$.

Expanding the factorials, we get $\frac{n(n-1)(n-2)(n-3)}{4!} = 126$.

Simplifying the left side, we have $\frac{n(n-1)(n-2)(n-3)}{24} = 126$.

Multiplying both sides by 24, we get $n(n-1)(n-2)(n-3) = 126 \times 24 = 3024$.

This is a quartic equation, and it's not easy to solve directly. However, we can guess and check to find the value of n.

By trial and error, we find that when n = 8, the left side of the equation becomes $8 \times 7 \times 6 \times 5 = 1680$, which is too large.

When n = 7, the left side of the equation becomes $7 \times 6 \times 5 \times 4 = 840$, which is also too large.

When n = 6, the left side of the equation becomes $6 \times 5 \times 4 \times 3 = 360$, which is still too large.

When n = 5, the left side of the equation becomes $5 \times 4 \times 3 \times 2 = 120$, which is still too small.

When n = 7, the left side of the equation becomes $7 \times 6 \times 5 \times 4 = 840$, which is too large.

When n = 8, the left side of the equation becomes $8 \times 7 \times 6 \times 5 = 1680$, which is too large.

When n = 9, the left side of the equation becomes $9 \times 8 \times 7 \times 6 = 3024$, which is exactly what we want.

So the positive integer n such that C(n,4) = 126 is n = 9.
The answer is: $9$"""


base_model_reference_prompt = '''
Calculate: Find the positive integer n such that C(n,4) = 126.

C(n,4) = n!/(4!(n-4)!) = 126
n(n-1)(n-2)(n-3)/24 = 126
n(n-1)(n-2)(n-3) = 3024
Try n = 9: 9×8×7×6 = ?'''


sft_model_reference_prompt = '''Find the positive integer n such that C(n,4) = 126.

Solve step by step:
1. Write C(n,4) in terms of factorials
2. Simplify the equation to find n(n-1)(n-2)(n-3)
3. Calculate what n(n-1)(n-2)(n-3) should equal
4. Try values systematically, starting with reasonable guesses
5. Find the exact value of n that satisfies the equation'''

