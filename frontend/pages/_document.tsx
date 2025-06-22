import { Html, Head, Main, NextScript } from 'next/document'

export default function Document() {
  return (
    <Html lang="de">
      <Head>
        <link rel="icon" href="/luota-current.jpeg" />
        <meta name="description" content="KI-gestützter Treuhand-Assistent für Finanzdaten und Dokumente" />
        <link
          rel="stylesheet"
          href="/highlight.css"
        />
        <link
          rel="stylesheet"
          href="/custom-math.css"
        />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  )
} 