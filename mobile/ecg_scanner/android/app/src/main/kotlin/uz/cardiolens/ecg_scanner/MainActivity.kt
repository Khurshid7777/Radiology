package uz.cardiolens.ecg_scanner

import android.os.Bundle
import android.view.WindowManager
import io.flutter.embedding.android.FlutterFragmentActivity

// FlutterFragmentActivity: local_auth needs it for the biometric prompt.
// FLAG_SECURE: no screenshots, screen recording or app-switcher snapshot of patient data
// (TZ addendum §18.9).
class MainActivity : FlutterFragmentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        window.setFlags(WindowManager.LayoutParams.FLAG_SECURE, WindowManager.LayoutParams.FLAG_SECURE)
        super.onCreate(savedInstanceState)
    }
}
