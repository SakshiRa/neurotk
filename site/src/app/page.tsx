export default function HomePage() {
  return (
    <main className="min-h-screen bg-white text-neutral-900">
      <div className="mx-auto max-w-4xl px-6 py-16 sm:py-20">
        <header className="space-y-6">
          <div className="space-y-4">
            <p className="text-5xl font-semibold tracking-tight text-neutral-900 sm:text-6xl">
              NeuroTK
            </p>
            <h1 className="text-3xl font-semibold leading-tight text-neutral-900 sm:text-4xl">
              Dataset validation and preprocessing for NIfTI studies
            </h1>
          </div>
          <p className="max-w-2xl text-base leading-relaxed text-neutral-600">
            NeuroTK provides deterministic dataset checks and optional preprocessing for neuroimaging workflows. It helps
            research teams audit input quality, standardize orientation and spacing, and export reproducible reports
            without changing original data.
          </p>
        </header>

        <section className="mt-14 border-t border-neutral-200 pt-10">
          <div className="grid gap-10 sm:grid-cols-3">
            <article className="space-y-4">
              <h2 className="text-lg font-semibold text-neutral-900">Validate</h2>
              <ul className="space-y-2 text-sm text-neutral-600">
                <li>File integrity and metadata checks</li>
                <li>Orientation and spacing consistency</li>
                <li>Missing label detection</li>
              </ul>
            </article>

            <article className="space-y-4">
              <h2 className="text-lg font-semibold text-neutral-900">Report</h2>
              <ul className="space-y-2 text-sm text-neutral-600">
                <li>JSON as source of truth</li>
                <li>Readable HTML summaries</li>
                <li>Deterministic, reproducible outputs</li>
              </ul>
            </article>

            <article className="space-y-4">
              <h2 className="text-lg font-semibold text-neutral-900">Preprocess</h2>
              <ul className="space-y-2 text-sm text-neutral-600">
                <li>Reorientation to canonical space</li>
                <li>Spacing normalization</li>
                <li>No data modification beyond transforms</li>
              </ul>
            </article>
          </div>
        </section>

        <section className="mt-12 border-t border-neutral-200 pt-10">
          <div className="space-y-8">
            <h2 className="text-xl font-semibold text-neutral-900">What NeuroTK Reports</h2>
            <div className="grid gap-8 sm:grid-cols-2">
              <div className="space-y-3">
                <h3 className="text-base font-semibold text-neutral-900">Dataset-level statistics</h3>
                <ul className="space-y-2 text-sm text-neutral-600">
                  <li>Image shape, spacing, and physical size summaries</li>
                  <li>Intensity distribution statistics across the dataset</li>
                  <li>Foreground and label coverage statistics</li>
                </ul>
              </div>
              <div className="space-y-3">
                <h3 className="text-base font-semibold text-neutral-900">Consistency and integrity checks</h3>
                <ul className="space-y-2 text-sm text-neutral-600">
                  <li>Orientation and spacing mismatches</li>
                  <li>Missing or partial labels</li>
                  <li>Metadata inconsistencies</li>
                </ul>
              </div>
              <div className="space-y-3">
                <h3 className="text-base font-semibold text-neutral-900">Reproducible outputs</h3>
                <ul className="space-y-2 text-sm text-neutral-600">
                  <li>Structured JSON reports (machine-readable)</li>
                  <li>Human-readable HTML summaries</li>
                  <li>Deterministic, audit-ready results</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        <footer className="mt-16 border-t border-neutral-200 pt-6 text-xs text-neutral-500">
          Research software for internal and academic use.
        </footer>
      </div>
    </main>
  );
}
