import Logo from './Logo'

const EXAMPLES = [
  'What are the fundamental rights?',
  'How is the Prime Minister appointed?',
  'What is the official language of Nepal?',
  'Is there a death penalty in Nepal?',
  'Which province is Kathmandu in?',
  'What does Article 17 say?',
]

export default function Welcome({ onPick }) {
  return (
    <section className="welcome">
      <div className="welcome-logo">
        <Logo size={44} />
      </div>
      <h1>Ask the Constitution of Nepal</h1>
      <p>
        Answers come only from the Constitution (2015, with the 2016 and 2020 amendments), and each
        one shows the Articles it is based on.
      </p>
      <div className="examples">
        {EXAMPLES.map((question) => (
          <button key={question} type="button" onClick={() => onPick(question)}>
            {question}
          </button>
        ))}
      </div>
    </section>
  )
}
