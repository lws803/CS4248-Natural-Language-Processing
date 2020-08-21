# Regex

* Regex can be considered as a pattern to specift text search strings to search a corupus of text
* Shows the *exact part of the string in a line that **first matches** the RE pattern.

## Examples
1. `woodchucks?` -> `woodchuck(s?)`: will match woodchuck or woodchucks
2. `[^A-Z]`: will match all characters that is not from A to Z. `^` will be treated as a normal
character if it is not the first character in s square bracket
3. `beg.n`: `.` represents a wildcard character. Any character can exist between beg or n.

# Finite state automata

* Regex is usually converted into a finite state automata before matching is performed.
* FSA accepts an input string if we run out of input and the FSA is in an <mark>accepting</mark> state

## State transition table

|   | b | a | ! |
|---|---|---|---|
| 0 | 1 | 0 | 0 |
| 1 | 0 | 2 | 0 |
| 2 | 0 | 3 | 0 |
| 3 | 0 | 3 | 4 |
| 4 | 0 | 0 | 0 |


## Deterministic FSA

* There must always be a transition to a new state, all states need to have an outgoing edge

### Formal language
* A set of strings
* A model which can both generate and recognize all and only the strings of a formal language

## Non-deterministic FSA

* Accepts an input string if there is <mark>at least some path</mark> in the NFSA that
leads to an accepting state and <mark>exhausting the input string</mark>.

### State space search

* Systematically search the space for all possible states
* Order of visiting each state is critical to the efficiency of the algorithm
    * DFS may lead to infinite loop
    * BFS requires large memory

## Deterministic FSA vs Non-deterministic FSA
DFSA is equivalent to NFSA

## Regular languages
The class of regular languages over Σ is formally defined as:

* Ø is a regular language
* for all a ⊆ Σ ∪ {ε}, {a} is a regular language
* If L1 and L2 are regular languages, then so are
    * L1 · L2 = {xy| x ⊆ L1, y ⊆ L2}, also known as the concatenation of L1 and L2
    * L1 ∪ L2
    * L1 *, L1 repeated 0 or more times
* regular languages are equivalent to FSA
