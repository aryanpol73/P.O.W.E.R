export default function Offline() {
  return (
    <div className="mx-auto mt-16 max-w-md text-center">
      <h1 className="text-lg font-semibold">You are offline</h1>
      <p className="mt-2 text-sm text-muted">
        Pages you opened recently still show their last data. Live forecasts and alerts resume as soon as the connection to the
        P.O.W.E.R. API is back.
      </p>
    </div>
  );
}
